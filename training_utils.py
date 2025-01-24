
import gc
import os
import sys
import numpy as np
import torch.backends
import torch.nn as nn
import time
from time import gmtime, strftime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
from manual_dataset import *
from rich.progress import Progress
from models import ResNet
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms
import random
eps = 1e-8


class RandomSpecAugHole:
    """
    Apply random hole masking to the spectrogram.
    """
    def __init__(self, num_holes=5, hole_height=10, hole_width=10, p=0.8):
        """
        :param num_holes: Number of random holes to apply.
        :param hole_height: Maximum height of each hole in frequency bins.
        :param hole_width: Maximum width of each hole in time steps.
        :param p: Probability of applying the SpecAugHole augmentations technique
        """
        self.num_holes = num_holes
        self.hole_height = hole_height
        self.hole_width = hole_width
        self.p = p

    def __call__(self, spectrogram):
        """
        Args:
            spectrogram (Tensor): Spectrogram of shape (channels, time, frequency).
        Returns:
            Tensor: Augmented spectrogram with random holes.
        """
        if np.random.rand() > self.p:
                return spectrogram
        else:
            cloned = spectrogram.clone()
            channels, time_steps, freq_bins = cloned.size()
            n_holes = random.randint(1, self.num_holes)

            for _ in range(n_holes):
                # Randomly choose hole size
                height = random.randint(1, self.hole_height)
                width = random.randint(1, self.hole_width)

                # Randomly choose top-left corner of the hole
                freq_start = random.randint(0, max(1, freq_bins - height))
                time_start = random.randint(0, max(1, time_steps - width))

                # Apply the mask
                cloned[:, time_start:time_start + width, freq_start:freq_start + height] = 0

            return cloned



class CosineWarmup_StepScheduler(_LRScheduler):
    """
    Learning rate scheduler that combines linear warmup with cosine annealing based on training steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_pct (float): Fraction of total steps for the linear warmup phase (e.g., 0.1 for 10%). Default is 0.05.
        total_steps (int): Total number of training steps.
        last_step (int, optional): The index of the last step. Default is -1.
    """
    
    def __init__(self, optimizer, warmup_pct: float = 0.05, 
                 total_steps: int = 10000, last_step: int = -1):
        if not 0.0 <= warmup_pct < 1.0:
            raise ValueError("warmup_pct must be in the range [0.0, 1.0).")
        if total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")
        
        self.warmup_steps = int(warmup_pct * total_steps)
        self.total_steps = total_steps
        super(CosineWarmup_StepScheduler, self).__init__(optimizer, last_step)

    def get_lr(self):
        """
        Computes the learning rate for the current step.

        Returns:
            list: Updated learning rates for each parameter group.
        """
        step = self.last_epoch + 1  # Increment step count
        lr_factors = [self.get_lr_factor(step) for _ in self.base_lrs]
        return [base_lr * factor for base_lr, factor in zip(self.base_lrs, lr_factors)]

    def get_lr_factor(self, step: int) -> float:
        """
        Computes the learning rate scaling factor based on the current step.

        Args:
            step (int): Current step number.

        Returns:
            float: Scaling factor for the learning rate.
        """
        if step < self.warmup_steps and self.warmup_steps != 0:
            # Linear warmup phase
            warmup_factor = step / self.warmup_steps
        else:
            warmup_factor = 1.0

        if step <= self.total_steps:
            # Cosine annealing phase
            progress = step / self.total_steps
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        else:
            # After total_steps, keep the learning rate at the minimum
            cosine_factor = 0.0

        return warmup_factor * cosine_factor

    def step(self, step: int = None):
        """
        Updates the learning rate. Should be called after each training step.

        Args:
            step (int, optional): The current step number. If not provided, increments internally.
        """
        if step is None:
            step = self.last_epoch + 1
        else:
            if step < 0:
                raise ValueError("step must be non-negative.")
            if step > self.total_steps:
                step = self.total_steps
        self.last_epoch = step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def count_parameters(model):
    """Returns the total number of parameters of the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def write_and_print(logger, out_string):
    """Write the string to the logging file and output it simulataenously"""
    try:
        logger.write(out_string+"\n")
        logger.flush()
        print(out_string, flush=True)
    except:
        print(datetime.now().strftime("%d%m%y_%H%M%S"))


def convert_output(predictions, n_classes = 3):
    
    predictions = predictions.detach().cpu().numpy()

    # --------------------------------------------------------------------------
    # 1) Flatten from (60, 10, 6) to (600, 6)
    # --------------------------------------------------------------------------
    def reshape_3Dto2D(A):
        return A.reshape(A.shape[0] * A.shape[1], A.shape[2])

    predictions = reshape_3Dto2D(predictions)  # shape -> (600, 6)
    
    # --------------------------------------------------------------------------
    # 2) Separate x and y for each of n_classes
    # --------------------------------------------------------------------------
    pred_x , pred_y = predictions[:, :n_classes] ,  predictions[:, n_classes:]
    
    # --------------------------------------------------------------------------
    # 3) SED mask : "Active" if sqrt(x^2 + y^2) > some_threshold
    # --------------------------------------------------------------------------
    sed = np.sqrt(pred_x ** 2 + pred_y ** 2) > 0.5

    # --------------------------------------------------------------------------
    # 4) Convert (x, y) -> azimuth in degrees
    #    arctan2(y, x) yields angle in radians in [-pi, pi].
    # --------------------------------------------------------------------------
    azi = np.arctan2(pred_y, pred_x) * 180.0 / np.pi   # shape (600, 3)
    
    # Put angles in [0, 360):
    azi[azi < 0] += 360.0
    
    converted_output = np.concatenate((sed, azi), axis=-1)
    # print(converted_output.shape)
    return converted_output


class ClassWiseMetrics:
    """
    Accumulates per-class SED accuracy, recall, and localization error
    for 3 classes over multiple batches.
    """
    def __init__(self, n_classes=3, threshold=0.5):
        self.n_classes = n_classes
        self.threshold = threshold

        # SED confusion counts per class
        self.tp = np.zeros(n_classes, dtype=np.int64)
        self.fp = np.zeros(n_classes, dtype=np.int64)
        self.tn = np.zeros(n_classes, dtype=np.int64)
        self.fn = np.zeros(n_classes, dtype=np.int64)

        # Localization error accumulators per class
        self.loc_err_sum = np.zeros(n_classes, dtype=np.float64)
        self.loc_err_count = np.zeros(n_classes, dtype=np.int64)

    def _wraparound_angle_diff_deg(self, a, b):
        """
        Compute absolute difference in degrees between arrays a and b,
        accounting for 360 wrap-around.
        """
        diff = np.abs(a - b) % 360
        return np.minimum(diff, 360 - diff)

    def update(self, gt, pred):
        """
        gt, pred: Tensors or numpy arrays of shape (batch_size, 6)
            [sed1, sed2, sed3, azi1, azi2, azi3].
        """
        # Move to CPU if necessary, and convert to numpy
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()

        # Separate SED from azimuth
        gt_sed  = gt[:, :self.n_classes]       # shape (B, 3)
        gt_azi  = gt[:, self.n_classes:self.n_classes*2]       # shape (B, 3)
        pred_sed = pred[:, :self.n_classes]    # shape (B, 3)
        pred_azi = pred[:, self.n_classes:self.n_classes*2]    # shape (B, 3)

        # Binarize predicted SED
        pred_sed_bin = (pred_sed > self.threshold).astype(np.int32)
        gt_sed_bin   = gt_sed.astype(np.int32)

        # Update confusion matrix stats for each class
        for i in range(self.n_classes):
            # Flatten for class i
            gt_i   = gt_sed_bin[:, i]
            pred_i = pred_sed_bin[:, i]

            # True Positive: both 1
            self.tp[i] += np.sum((gt_i == 1) & (pred_i == 1))
            # False Positive: gt=0, pred=1
            self.fp[i] += np.sum((gt_i == 0) & (pred_i == 1))
            # True Negative: both 0
            self.tn[i] += np.sum((gt_i == 0) & (pred_i == 0))
            # False Negative: gt=1, pred=0
            self.fn[i] += np.sum((gt_i == 1) & (pred_i == 0))

            # Localization error only for matched positives
            matched_mask = (gt_i == 1) & (pred_i == 1)
            if np.any(matched_mask):
                diff_deg = self._wraparound_angle_diff_deg(pred_azi[matched_mask, i],
                                                           gt_azi[matched_mask, i])
                self.loc_err_sum[i]   += diff_deg.sum()
                self.loc_err_count[i] += diff_deg.size

    def compute(self):
        """
        Returns dictionary of per-class metrics:
          - accuracy[i]
          - recall[i]
          - loc_error[i]
        """
        accuracy = []
        recall = []
        loc_error = []

        for i in range(self.n_classes):
            # Accuracy_i = (TP + TN) / total_samples_for_class
            # But note "total_samples_for_class" is actually (TP + TN + FP + FN).
            total_i = self.tp[i] + self.tn[i] + self.fp[i] + self.fn[i]
            if total_i > 0:
                accuracy_i = (self.tp[i] + self.tn[i]) / float(total_i)
            else:
                accuracy_i = np.nan

            # Recall_i = TP / (TP + FN)
            denom_recall_i = (self.tp[i] + self.fn[i])
            if denom_recall_i > 0:
                recall_i = self.tp[i] / float(denom_recall_i)
            else:
                recall_i = np.nan

            # Localization error = average angle diff for matched positives
            if self.loc_err_count[i] > 0:
                loc_err_i = self.loc_err_sum[i] / float(self.loc_err_count[i])
            else:
                loc_err_i = np.nan

            accuracy.append(accuracy_i)
            recall.append(recall_i)
            loc_error.append(loc_err_i)

        return {
            'accuracy':  accuracy,   # list of length 3
            'recall':    recall,     # list of length 3
            'loc_error': loc_error,  # list of length 3
        }


def wraparound_azimuth_diff_deg(az1, az2):
    """
    Compute absolute difference between two azimuth angles in [0..360],
    taking into account wrap-around. Returns a value in [0..180].
    """
    diff = np.abs(az1 - az2) % 360
    diff = np.minimum(diff, 360 - diff)
    return diff


class SELDMetricsAzimuth:
    """
    Accumulator for 3-class SELD where each sample is:
        [SED1, SED2, SED3, AZI1, AZI2, AZI3]

    We'll accumulate over an entire epoch (or multiple epochs) and
    compute final ER, F, LE, LR at the end.
    """

    def __init__(self, n_classes=3, azimuth_threshold=20.0, sed_threshold=0.5):
        self.n_classes = n_classes
        self.azimuth_threshold = azimuth_threshold
        self.sed_threshold = sed_threshold

        # Location-sensitive detection accumulators
        self.TP = 0
        self.FP = 0
        self.FN = 0
        
        # For SED-like error-rate breakdown
        self.S = 0  # Substitution
        self.D = 0  # Deletion
        self.I = 0  # Insertion
        self.Nref = 0  # total reference events

        # Class-sensitive localization accumulators
        self.total_DE = 0.0  # sum of azimuth differences for matched TPs
        self.DE_TP = 0
        self.DE_FP = 0
        self.DE_FN = 0

    def update(self, gt, pred):
        """
        Args:
            gt, pred: Tensors or numpy arrays of shape (batch_size, 6)
                      [sed1, sed2, sed3, azi1, azi2, azi3]
        We parse:
          - sed = columns [0..3)
          - azi = columns [3..6)

        Then we:
          1) Binarize predicted SED with self.sed_threshold
          2) Compare each class's GT SED vs. predicted SED
          3) For matched positives, measure azimuth difference
             and decide if it's within azimuth_threshold for a "true positive"
             or a "false positive."

        We'll do a mini confusion-matrix update for each sample.
        """

        # 1) Move to CPU np arrays if needed
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()

        # 2) Separate SED from azimuth
        gt_sed  = gt[:, :self.n_classes]      # shape (B, 3)
        gt_azi  = gt[:, self.n_classes:2*self.n_classes]  # shape (B, 3)
        pred_sed = pred[:, :self.n_classes]   # shape (B, 3)
        pred_azi = pred[:, self.n_classes:2*self.n_classes]  # shape (B, 3)

        # 3) Binarize predicted SED
        pred_sed_bin = (pred_sed > self.sed_threshold).astype(np.int32)
        gt_sed_bin   = gt_sed.astype(np.int32)

        batch_size = gt.shape[0]

        # Loop over each sample in the batch
        for i in range(batch_size):
            # For SED-like error breakdown on a "per-class" basis,
            # let's see how many GT=1, pred=1, etc.
            # Then we can find how many TPs, FPs, FNs for location-sensitivity.
            loc_FN_sample = 0
            loc_FP_sample = 0

            # Count how many ground-truth events are in sample i
            n_ref_events = np.sum(gt_sed_bin[i])  # e.g. could be 0..3
            self.Nref += n_ref_events  # used in ER denominator

            # For each class c in [0..2]
            for c in range(self.n_classes):
                gt_active = (gt_sed_bin[i, c] == 1)
                pred_active = (pred_sed_bin[i, c] == 1)

                if gt_active and pred_active:
                    # This is a matched "positive" at the SED level.
                    # Now check azimuth difference for location sensitivity
                    diff = wraparound_azimuth_diff_deg(pred_azi[i,c], gt_azi[i,c])
                    
                    # Accumulate difference for class-sensitive localization
                    self.total_DE += diff
                    self.DE_TP += 1  # we found a matched class

                    if diff <= self.azimuth_threshold:
                        self.TP += 1
                    else:
                        loc_FP_sample += 1
                        self.FP += 1

                elif gt_active and not pred_active:
                    # missed event => false negative
                    loc_FN_sample += 1
                    self.FN += 1
                    self.DE_FN += 1

                elif (not gt_active) and pred_active:
                    # spurious event => false positive
                    loc_FP_sample += 1
                    self.FP += 1
                    self.DE_FP += 1
                else:
                    # both inactive => true negative for SED, doesn't affect location metrics
                    pass

            # After analyzing all 3 classes for this sample,
            # we can update the S, D, I for error rate:
            self.S += min(loc_FP_sample, loc_FN_sample)  # substitution
            self.D += max(0, loc_FN_sample - loc_FP_sample)  # deletion
            self.I += max(0, loc_FP_sample - loc_FN_sample)  # insertion

    def compute(self):
        """
        Returns the final SELD metrics after all updates:
          ER, F, LE, LR
        """
        # 1) Location-sensitive detection metrics
        ER = (self.S + self.D + self.I) / float(self.Nref + eps)
        F  = self.TP / (eps + self.TP + 0.5 * (self.FP + self.FN))

        # 2) Class-sensitive localization metrics
        if self.DE_TP > 0:
            LE = self.total_DE / float(self.DE_TP)
        else:
            LE = 180.0  # fallback if no TPs

        LR = self.DE_TP / (eps + self.DE_TP + self.DE_FN)

        return ER, F, LE, LR


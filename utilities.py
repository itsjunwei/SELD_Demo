import numpy as np
import torch.backends
from time import gmtime, strftime
import torch
from datetime import datetime
from torch.optim.lr_scheduler import _LRScheduler
import random
eps = np.finfo(float).eps


# --------------------------------------------------------------------------
# Learning Rate Scheduler
# --------------------------------------------------------------------------

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



class DecayScheduler(_LRScheduler):
    """
    Decays the learning rate by a fixed factor every specified number of epochs

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        decay_factor (float, optional): Factor by which to decay the learning rate. Default is 0.98.
        min_lr (float, optional): Minimum learning rate. Default is 1e-5.
        nb_epoch_to_decay (int, optional): Number of epochs between each decay step. Default is 2.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
    """

    def __init__(self, optimizer, decay_factor: float = 0.98, min_lr: float = 1e-5, 
                 last_epoch: int = -1, nb_epoch_to_decay: int = 2):
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.nb_epoch_to_decay = nb_epoch_to_decay
        super(DecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        if epoch < 1:
            # Return the initial learning rates for the first epoch
            return self.base_lrs
        else:
            # Calculate how many decay steps have occurred
            decay_steps = epoch // self.nb_epoch_to_decay
            # Compute the decay factor based on the number of decay steps
            current_decay = self.decay_factor ** decay_steps
            # Apply decay to each base learning rate, ensuring it doesn't go below min_lr
            return [max(base_lr * current_decay, self.min_lr) for base_lr in self.base_lrs]

# --------------------------------------------------------------------------
# Misc Utility Functions
# --------------------------------------------------------------------------

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
        
def wraparound_azimuth_diff_deg(az1, az2):
    """
    Compute absolute difference between two azimuth angles in [0..360],
    taking into account wrap-around. Returns a value in [0..180].
    """
    diff = np.abs(az1 - az2) % 360
    diff = np.minimum(diff, 360 - diff)
    return diff


def convert_output(predictions, n_classes = 3, sed_threshold=0.5):
    
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
    sed = np.sqrt(pred_x ** 2 + pred_y ** 2) > sed_threshold

    # --------------------------------------------------------------------------
    # 4) Convert (x, y) -> azimuth in degrees
    #    arctan2(y, x) yields angle in radians in [-pi, pi].
    # --------------------------------------------------------------------------
    azi = np.arctan2(pred_y, pred_x) * 180.0 / np.pi   # shape (600, 3)
    azi = azi * sed
    
    # Put angles in [0, 360):
    azi[azi < 0] += 360.0
    
    converted_output = np.concatenate((sed, azi), axis=-1)
    return converted_output


# --------------------------------------------------------------------------
# SELD Metrics
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Data Augmentation Techniques
# --------------------------------------------------------------------------

class ComposeTransformNp:
    """
    Compose a list of data augmentation on numpy array.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray):
        for transform in self.transforms:
            x = transform(x)
        return x

class DataAugmentNumpyBase:
    """
    Base class for data augmentation for audio spectrogram of numpy array. This class does not alter label
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray):
        if self.always_apply:
            return self.apply(x)
        else:
            if np.random.rand() < self.p:
                return self.apply(x)
            else:
                return x

    def apply(self, x: np.ndarray):
        raise NotImplementedError


class CompositeCutout(DataAugmentNumpyBase):
    """
    This data augmentation combine Random cutout, specaugment, cutout hole.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.random_cutout = RandomCutoutNp(always_apply=True, image_aspect_ratio=image_aspect_ratio,
                                            n_zero_channels=n_zero_channels,
                                            is_filled_last_channels=is_filled_last_channels)
        self.spec_augment = SpecAugmentNp(always_apply=True, n_zero_channels=n_zero_channels,
                                          is_filled_last_channels=is_filled_last_channels)
        self.random_cutout_hole = RandomCutoutHoleNp(always_apply=True, n_zero_channels=n_zero_channels,
                                                     is_filled_last_channels=is_filled_last_channels)

    def apply(self, x: np.ndarray):
        choice = np.random.randint(0, 3, 1)[0]
        if choice == 0:
            return self.random_cutout(x)
        elif choice == 1:
            return self.spec_augment(x)
        elif choice == 2:
            return self.random_cutout_hole(x)


class RandomCutoutNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a rectangular area from the input image. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 random_value: float = None, n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param image_aspect_ratio: height/width ratio. For spectrogram: n_time_steps/ n_features.
        :param random_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.random_value = random_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels
        # Params: s: area, r: height/width ratio.
        self.s_l = 0.02
        self.s_h = 0.3
        self.r_1 = 0.3
        self.r_2 = 1 / 0.3
        if image_aspect_ratio > 1:
            self.r_1 = self.r_1 * image_aspect_ratio
        elif image_aspect_ratio < 1:
            self.r_2 = self.r_2 * image_aspect_ratio

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features) or (n_time_steps, n_features)>: input spectrogram.
        :return: random cutout x
        """
        # get image size
        image_dim = x.ndim
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        # Initialize output
        output_img = x.copy()
        # random erase
        s = np.random.uniform(self.s_l, self.s_h) * img_h * img_w
        r = np.random.uniform(self.r_1, self.r_2)
        w = np.min((int(np.sqrt(s / r)), img_w - 1))
        h = np.min((int(np.sqrt(s * r)), img_h - 1))
        left = np.random.randint(0, img_w - w)
        top = np.random.randint(0, img_h - h)
        if self.random_value is None:
            c = np.random.uniform(min_value, max_value)
        else:
            c = self.random_value
        if image_dim == 2:
            output_img[top:top + h, left:left + w] = c
        else:
            if self.n_zero_channels is None:
                output_img[:, top:top + h, left:left + w] = c
            else:
                output_img[:-self.n_zero_channels,  top:top + h, left:left + w] = c
                if self.is_filled_last_channels:
                    output_img[-self.n_zero_channels:, top:top + h, left:left + w] = 0.0

        return output_img


class SpecAugmentNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly remove horizontal or vertical strips from image. Tested
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, time_max_width: int = None,
                 freq_max_width: int = None, n_time_stripes: int = 1, n_freq_stripes: int = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param time_max_width: maximum time width to remove.
        :param freq_max_width: maximum freq width to remove.
        :param n_time_stripes: number of time stripes to remove.
        :param n_freq_stripes: number of freq stripes to remove.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.time_max_width = time_max_width
        self.freq_max_width = freq_max_width
        self.n_time_stripes = n_time_stripes
        self.n_freq_stripes = n_freq_stripes
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        n_frames = x.shape[1]
        n_freqs = x.shape[2]
        min_value = np.min(x)
        max_value = np.max(x)
        if self.time_max_width is None:
            time_max_width = int(0.15 * n_frames)
        else:
            time_max_width = self.time_max_width
        time_max_width = np.max((1, time_max_width))
        if self.freq_max_width is None:
            freq_max_width = int(0.2 * n_freqs)
        else:
            freq_max_width = self.freq_max_width
        freq_max_width = np.max((1, freq_max_width))

        new_spec = x.copy()

        for i in np.arange(self.n_time_stripes):
            dur = np.random.randint(1, time_max_width, 1)[0]
            start_idx = np.random.randint(0, n_frames - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            if self.n_zero_channels is None:
                new_spec[:, start_idx:start_idx + dur, :] = random_value
            else:
                new_spec[:-self.n_zero_channels, start_idx:start_idx + dur, :] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, start_idx:start_idx + dur, :] = 0.0

        for i in np.arange(self.n_freq_stripes):
            dur = np.random.randint(1, freq_max_width, 1)[0]
            start_idx = np.random.randint(0, n_freqs - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            if self.n_zero_channels is None:
                new_spec[:, :, start_idx:start_idx + dur] = random_value
            else:
                new_spec[:-self.n_zero_channels, :, start_idx:start_idx + dur] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, :, start_idx:start_idx + dur] = 0.0

        return new_spec

class RandomCutoutHoleNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a few small holes in the spectrogram. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_max_holes: int = 8, max_h_size: int = 8,
                 max_w_size: int = 8, filled_value: float = None, n_zero_channels: int = None,
                 is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param n_max_holes: Maximum number of holes to cutout.
        :param max_h_size: Maximum time frames of the cutout holes.
        :param max_w_size: Maximum freq bands of the cutout holes.
        :param filled_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.n_max_holes = n_max_holes
        self.max_h_size = np.max((max_h_size, 5))
        self.max_w_size = np.max((max_w_size, 5))
        self.filled_value = filled_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def apply(self, x: np.ndarray):
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        new_spec = x.copy()
        # n_cutout_holes = np.random.randint(1, self.n_max_holes, 1)[0]
        n_cutout_holes = self.n_max_holes
        for ihole in np.arange(n_cutout_holes):
            # w = np.random.randint(4, self.max_w_size, 1)[0]
            # h = np.random.randint(4, self.max_h_size, 1)[0]
            w = self.max_w_size
            h = self.max_h_size
            left = np.random.randint(0, img_w - w)
            top = np.random.randint(0, img_h - h)
            if self.filled_value is None:
                filled_value = np.random.uniform(min_value, max_value)
            else:
                filled_value = self.filled_value
            if self.n_zero_channels is None:
                new_spec[:, top:top + h, left:left + w] = filled_value
            else:
                new_spec[:-self.n_zero_channels, top:top + h, left:left + w] = filled_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, top:top + h, left:left + w] = 0.0

        return new_spec

class RandomShiftUpDownNp(DataAugmentNumpyBase):
    """
    This data augmentation random shift the spectrogram up or down.
    """
    def __init__(self, always_apply=False, p=0.5, freq_shift_range: int = None, direction: str = None, mode='reflect',
                 n_last_channels: int = 0):
        super().__init__(always_apply, p)
        self.freq_shift_range = freq_shift_range
        self.direction = direction
        self.mode = mode
        self.n_last_channels = n_last_channels

    def apply(self, x: np.ndarray):
        if self.always_apply is False:
            return x
        else:
            if np.random.rand() < self.p:
                return x
            else:
                n_channels, n_timesteps, n_features = x.shape
                if self.freq_shift_range is None:
                    self.freq_shift_range = int(n_features * 0.08)
                shift_len = np.random.randint(1, self.freq_shift_range, 1)[0]
                if self.direction is None:
                    direction = np.random.choice(['up', 'down'], 1)[0]
                else:
                    direction = self.direction
                new_spec = x.copy()
                if self.n_last_channels == 0:
                    if direction == 'up':
                        new_spec = np.pad(new_spec, ((0, 0), (0, 0), (shift_len, 0)), mode=self.mode)[:, :, 0:n_features]
                    else:
                        new_spec = np.pad(new_spec, ((0, 0), (0, 0), (0, shift_len)), mode=self.mode)[:, :, shift_len:]
                else:
                    if direction == 'up':
                        new_spec[:-self.n_last_channels] = np.pad(
                            new_spec[:-self.n_last_channels], ((0, 0), (0, 0), (shift_len, 0)), mode=self.mode)[:, :, 0:n_features]
                    else:
                        new_spec[:-self.n_last_channels] = np.pad(
                            new_spec[:-self.n_last_channels], ((0, 0), (0, 0), (0, shift_len)), mode=self.mode)[:, :, shift_len:]
                return new_spec

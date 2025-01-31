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


class SELDMetricsAzimuth:
    """
    Accumulator for 3-class SELD where each sample is:
        [SED1, SED2, SED3, AZI1, AZI2, AZI3]

    We'll accumulate over an entire epoch (or multiple epochs) and
    compute final ER, F, LE, LR at the end, **and also class-wise metrics**.
    """

    def __init__(self, n_classes=3, azimuth_threshold=20.0, sed_threshold=0.5, out_class=False):
        self.n_classes = n_classes
        self.azimuth_threshold = azimuth_threshold
        self.sed_threshold = sed_threshold
        self.out_class = out_class

        # -----------------------
        # Overall accumulators
        # -----------------------
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
        self.DE_TP = 0       # TPs at SED level that were matched for location
        self.DE_FP = 0
        self.DE_FN = 0

        # -----------------------
        # Class-wise accumulators
        # -----------------------
        # For each class, store SED-level TPs, FPs, FNs,
        # error-rate breakdown, and location metrics.
        self.class_TP = np.zeros(self.n_classes, dtype=int)
        self.class_FP = np.zeros(self.n_classes, dtype=int)
        self.class_FN = np.zeros(self.n_classes, dtype=int)

        self.class_S = np.zeros(self.n_classes, dtype=int)
        self.class_D = np.zeros(self.n_classes, dtype=int)
        self.class_I = np.zeros(self.n_classes, dtype=int)
        self.class_Nref = np.zeros(self.n_classes, dtype=int)

        # For localization
        self.class_total_DE = np.zeros(self.n_classes, dtype=float)
        self.class_DE_TP = np.zeros(self.n_classes, dtype=int)
        self.class_DE_FP = np.zeros(self.n_classes, dtype=int)
        self.class_DE_FN = np.zeros(self.n_classes, dtype=int)


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
        """

        # 1) Move to CPU np arrays if needed
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()

        # 2) Separate SED from azimuth
        gt_sed   = gt[:, :self.n_classes]                   # shape (B, 3)
        gt_azi   = gt[:, self.n_classes:2*self.n_classes]   # shape (B, 3)
        pred_sed = pred[:, :self.n_classes]                 # shape (B, 3)
        pred_azi = pred[:, self.n_classes:2*self.n_classes] # shape (B, 3)

        # 3) Binarize predicted SED
        pred_sed_bin = (pred_sed > self.sed_threshold).astype(np.int32)
        gt_sed_bin   = gt_sed.astype(np.int32)

        batch_size = gt.shape[0]

        for i in range(batch_size):
            # Count total # of reference events (for overall metrics)
            n_ref_events = np.sum(gt_sed_bin[i])  # e.g. could be 0..3
            self.Nref += n_ref_events

            # We'll track how many false positives and negatives
            # occurred within this sample across classes,
            # for computing Substitution, Deletion, Insertion
            loc_FN_sample = 0
            loc_FP_sample = 0

            # For each class c in [0..n_classes-1]
            for c in range(self.n_classes):
                gt_active = (gt_sed_bin[i, c] == 1)
                pred_active = (pred_sed_bin[i, c] == 1)

                # Update the total reference events for that class
                if gt_active:
                    self.class_Nref[c] += 1

                if gt_active and pred_active:
                    # This is a matched "positive" at the SED level.
                    # Now check azimuth difference for location sensitivity
                    diff = wraparound_azimuth_diff_deg(pred_azi[i, c], gt_azi[i, c])

                    # Accumulate difference for class-sensitive localization
                    self.total_DE += diff
                    self.class_total_DE[c] += diff

                    self.DE_TP += 1
                    self.class_DE_TP[c] += 1

                    # Decide if it's within localization threshold => "true positive"
                    if diff <= self.azimuth_threshold:
                        self.TP += 1
                        self.class_TP[c] += 1
                    else:
                        self.FP += 1
                        self.class_FP[c] += 1
                        self.DE_FP += 1
                        self.class_DE_FP[c] += 1
                        loc_FP_sample += 1

                elif gt_active and not pred_active:
                    # missed event => false negative
                    self.FN += 1
                    self.class_FN[c] += 1

                    self.DE_FN += 1
                    self.class_DE_FN[c] += 1

                    loc_FN_sample += 1

                elif (not gt_active) and pred_active:
                    # spurious event => false positive
                    self.FP += 1
                    self.class_FP[c] += 1

                    self.DE_FP += 1
                    self.class_DE_FP[c] += 1

                    loc_FP_sample += 1
                else:
                    # both inactive => true negative for SED, no location metrics update
                    pass

            # After analyzing all n_classes for this sample,
            # update S, D, I for the overall error rate:
            self.S += min(loc_FP_sample, loc_FN_sample)
            self.D += max(0, loc_FN_sample - loc_FP_sample)
            self.I += max(0, loc_FP_sample - loc_FN_sample)

    def compute(self):
        """
        Returns the final SELD metrics after all updates, for the entire set:
          ER, F, LE, LR
        And also returns the class-wise metrics in a dict or tuple.
        """

        # ---- Overall metrics ----
        ER = (self.S + self.D + self.I) / float(self.Nref + eps)
        F  = self.TP / (eps + self.TP + 0.5 * (self.FP + self.FN))

        if self.DE_TP > 0:
            LE = self.total_DE / float(self.DE_TP)
        else:
            LE = 180.0  # fallback if no TPs

        LR = self.DE_TP / (eps + self.DE_TP + self.DE_FN)

        # ---- Class-wise metrics ----
        if self.out_class:
            class_ER = np.zeros(self.n_classes, dtype=np.float32)
            class_F  = np.zeros(self.n_classes, dtype=np.float32)
            class_LE = np.zeros(self.n_classes, dtype=np.float32)
            class_LR = np.zeros(self.n_classes, dtype=np.float32)

            for c in range(self.n_classes):
                # Error Rate (ER) for class c
                class_ER[c] = (
                    (self.class_FP[c] + self.class_FN[c]) /
                    float(self.class_Nref[c] + eps)
                )

                # F-score for class c
                class_F[c] = (
                    self.class_TP[c] /
                    (eps + self.class_TP[c] + 0.5 * (self.class_FP[c] + self.class_FN[c]))
                )

                # Localization metrics for class c
                if self.class_DE_TP[c] > 0:
                    class_LE[c] = self.class_total_DE[c] / float(self.class_DE_TP[c])
                else:
                    class_LE[c] = 180.0  # fallback if no TPs for class c

                class_LR[c] = (
                    self.class_DE_TP[c] /
                    (eps + self.class_DE_TP[c] + self.class_DE_FN[c])
                )

            # Return both overall and class-wise
            classwise = {
                'ER': class_ER,
                'F':  class_F,
                'LE': class_LE,
                'LR': class_LR
            }
            
            class_names = ["Alarm", "Impact", "Speech"]
            print("Class-wise Metrics:")
            print(f"{'Class':<10} {'ER':>7} {'F':>7} {'LE':>7} {'LR':>7}")
            for i, cname in enumerate(class_names):
                er_i  = classwise['ER'][i]
                f_i   = classwise['F'][i]
                le_i  = classwise['LE'][i]
                lr_i  = classwise['LR'][i]
                print(f"{cname:<10} {er_i:7.3f} {f_i:7.3f} {le_i:7.3f} {lr_i:7.3f}")

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



class CompositeCutout:
    """
    This data augmentation combines Random Cutout, SpecAugment, and Random Cutout Hole.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transformations regardless of probability.
        :param p: Probability to apply transformations if always_apply is False.
        :param image_aspect_ratio: Aspect ratio for Random Cutout.
        :param n_zero_channels: If given, these last n_zero_channels will be filled with zeros instead of random values.
        :param is_filled_last_channels: If False, does not fill n_zero_channels with zeros.
        """
        self.always_apply = always_apply
        self.p = p
        self.random_cutout = RandomCutoutTensor(
            always_apply=True,
            image_aspect_ratio=image_aspect_ratio,
            n_zero_channels=n_zero_channels,
            is_filled_last_channels=is_filled_last_channels
        )
        self.spec_augment = SpecAugmentTensor(
            always_apply=True,
            n_zero_channels=n_zero_channels,
            is_filled_last_channels=is_filled_last_channels
        )
        self.random_cutout_hole = RandomCutoutHoleTensor(
            always_apply=True,
            n_zero_channels=n_zero_channels,
            is_filled_last_channels=is_filled_last_channels
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one of the augmentation techniques randomly.
        
        :param x: Input tensor of shape (n_channels, n_time_steps, n_features) or (n_time_steps, n_features).
        :return: Augmented tensor.
        """
        if not self.always_apply and torch.rand(1).item() > self.p:
            return x

        choice = torch.randint(0, 3, (1,)).item()
        if choice == 0:
            return self.random_cutout(x)
        elif choice == 1:
            return self.spec_augment(x)
        elif choice == 2:
            return self.random_cutout_hole(x)
        else:
            return x  # Fallback in case of unexpected choice




class RandomCutoutTensor:
    """
    This data augmentation randomly cuts out a rectangular area from the input tensor.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 random_value: float = None, n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is False, p is the probability to apply transform.
        :param image_aspect_ratio: Height/width ratio. For spectrogram: n_time_steps / n_features.
        :param random_value: Random value to fill in the cutout area. If None, randomly fill the cutout area with value
                             between min and max of input.
        :param n_zero_channels: If given, these last n_zero_channels will be filled with zeros instead of random values.
        :param is_filled_last_channels: If False, does not cutout n_zero_channels.
        """
        self.always_apply = always_apply
        self.p = p
        self.random_value = random_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

        # Parameters for area and aspect ratio
        self.s_l = 0.02
        self.s_h = 0.3
        self.r_1 = 0.3
        self.r_2 = 1 / 0.3
        if image_aspect_ratio > 1:
            self.r_1 *= image_aspect_ratio
        elif image_aspect_ratio < 1:
            self.r_2 *= image_aspect_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.always_apply and torch.rand(1).item() > self.p:
            return x

        # Get image size
        image_dim = x.dim()
        img_h, img_w = x.shape[-2], x.shape[-1]

        min_value = torch.min(x)
        max_value = torch.max(x)

        # Initialize output
        output_img = x.clone()

        # Random erase
        s = torch.empty(1).uniform_(self.s_l, self.s_h).item() * img_h * img_w
        r = torch.empty(1).uniform_(self.r_1, self.r_2).item()
        w = min(int(torch.sqrt(torch.tensor(s / r)).item()), img_w - 1)
        h = min(int(torch.sqrt(torch.tensor(s * r)).item()), img_h - 1)

        if img_w - w <= 0 or img_h - h <= 0:
            # If the calculated width or height is too large, skip cutout
            return output_img

        left = torch.randint(0, img_w - w, (1,)).item()
        top = torch.randint(0, img_h - h, (1,)).item()

        if self.random_value is None:
            c = torch.empty(1).uniform_(min_value, max_value).item()
        else:
            c = self.random_value

        if image_dim == 2:
            output_img[top:top + h, left:left + w] = c
        else:
            if self.n_zero_channels is None:
                output_img[:, top:top + h, left:left + w] = c
            else:
                output_img[:-self.n_zero_channels, top:top + h, left:left + w] = c
                if self.is_filled_last_channels:
                    output_img[-self.n_zero_channels:, top:top + h, left:left + w] = 0.0

        return output_img



class SpecAugmentTensor:
    """
    This data augmentation randomly removes horizontal or vertical strips from the input tensor.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, time_max_width: int = None,
                 freq_max_width: int = None, n_time_stripes: int = 1, n_freq_stripes: int = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is False, p is the probability to apply transform.
        :param time_max_width: Maximum time width to remove.
        :param freq_max_width: Maximum frequency width to remove.
        :param n_time_stripes: Number of time stripes to remove.
        :param n_freq_stripes: Number of frequency stripes to remove.
        :param n_zero_channels: If given, these last n_zero_channels will be filled with zeros instead of random values.
        :param is_filled_last_channels: If False, does not cutout n_zero_channels.
        """
        self.always_apply = always_apply
        self.p = p
        self.time_max_width = time_max_width
        self.freq_max_width = freq_max_width
        self.n_time_stripes = n_time_stripes
        self.n_freq_stripes = n_freq_stripes
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.always_apply and torch.rand(1).item() > self.p:
            return x

        # Ensure input has 3 dimensions
        assert x.dim() == 3, 'Error: dimension of input spectrogram is not 3!'

        n_channels, n_frames, n_freqs = x.shape
        min_value = torch.min(x)
        max_value = torch.max(x)

        # Determine time and frequency widths
        time_max_width = int(0.15 * n_frames) if self.time_max_width is None else self.time_max_width
        time_max_width = max(1, time_max_width)

        freq_max_width = int(0.2 * n_freqs) if self.freq_max_width is None else self.freq_max_width
        freq_max_width = max(1, freq_max_width)

        new_spec = x.clone()

        # Apply time stripes
        for _ in range(self.n_time_stripes):
            dur = torch.randint(1, time_max_width + 1, (1,)).item()
            if n_frames - dur <= 0:
                continue  # Skip if duration is too large
            start_idx = torch.randint(0, n_frames - dur + 1, (1,)).item()
            random_value = torch.empty(1).uniform_(min_value, max_value).item()

            if self.n_zero_channels is None:
                new_spec[:, start_idx:start_idx + dur, :] = random_value
            else:
                new_spec[:-self.n_zero_channels, start_idx:start_idx + dur, :] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, start_idx:start_idx + dur, :] = 0.0

        # Apply frequency stripes
        for _ in range(self.n_freq_stripes):
            dur = torch.randint(1, freq_max_width + 1, (1,)).item()
            if n_freqs - dur <= 0:
                continue  # Skip if duration is too large
            start_idx = torch.randint(0, n_freqs - dur + 1, (1,)).item()
            random_value = torch.empty(1).uniform_(min_value, max_value).item()

            if self.n_zero_channels is None:
                new_spec[:, :, start_idx:start_idx + dur] = random_value
            else:
                new_spec[:-self.n_zero_channels, :, start_idx:start_idx + dur] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, :, start_idx:start_idx + dur] = 0.0

        return new_spec



class RandomCutoutHoleTensor:
    """
    This data augmentation randomly cuts out a few small holes in the input tensor.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_max_holes: int = 8, max_h_size: int = 8,
                 max_w_size: int = 8, filled_value: float = None, n_zero_channels: int = None,
                 is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is False, p is the probability to apply transform.
        :param n_max_holes: Maximum number of holes to cutout.
        :param max_h_size: Maximum time frames of the cutout holes.
        :param max_w_size: Maximum frequency bands of the cutout holes.
        :param filled_value: Random value to fill in the cutout area. If None, randomly fill the cutout area with value
                             between min and max of input.
        :param n_zero_channels: If given, these last n_zero_channels will be filled with zeros instead of random values.
        :param is_filled_last_channels: If False, does not cutout n_zero_channels.
        """
        self.always_apply = always_apply
        self.p = p
        self.n_max_holes = n_max_holes
        self.max_h_size = max(max_h_size, 5)
        self.max_w_size = max(max_w_size, 5)
        self.filled_value = filled_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.always_apply and torch.rand(1).item() > self.p:
            return x

        # Ensure input has 3 dimensions
        assert x.dim() == 3, 'Error: dimension of input spectrogram is not 3!'

        n_channels, img_h, img_w = x.shape
        min_value = torch.min(x)
        max_value = torch.max(x)

        new_spec = x.clone()

        n_cutout_holes = self.n_max_holes  # You can randomize this if desired

        for _ in range(n_cutout_holes):
            w = self.max_w_size  # Alternatively, use torch.randint(4, self.max_w_size + 1, (1,)).item()
            h = self.max_h_size  # Alternatively, use torch.randint(4, self.max_h_size + 1, (1,)).item()

            if img_w - w <= 0 or img_h - h <= 0:
                continue  # Skip if hole size is too large

            left = torch.randint(0, img_w - w + 1, (1,)).item()
            top = torch.randint(0, img_h - h + 1, (1,)).item()

            if self.filled_value is None:
                filled_value = torch.empty(1).uniform_(min_value, max_value).item()
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

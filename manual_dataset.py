"""
This module load all seld feature into memory.
Reference code:  https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization
Note: there are two frame rates:
    1. feature frame rate: 80 frames/s
    2. label frame rate: 10 frames/s
"""
from rich.progress import Progress
import os
from typing import List
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class seldDatabase:
    def __init__(self,
                 feat_label_dir : str = './feat_label', 
                 n_classes: int = 3, fs: int = 24000,
                 n_fft: int = 512, hop_len: int = 300, label_rate: float = 10, 
                 train_chunk_len_s: float = 1.0, train_chunk_hop_len_s: float = 0.5,
                 n_channels: int = 7, n_bins: int = 191):

        self.feat_label_dir = feat_label_dir
        self.feature_train_dir = os.path.join(feat_label_dir, "train", "tracks")
        self.feature_test_dir  = os.path.join(feat_label_dir, "test", "metadata")
        self.gt_meta_train_dir = os.path.join(feat_label_dir, "train", "tracks")
        self.gt_meta_test_dir  = os.path.join(feat_label_dir, "test", "metadata")
        print("Loading files from {}\n\t\t and {}".format(self.feature_train_dir, self.gt_meta_train_dir))

        self.n_classes = n_classes
        self.fs = fs
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.label_rate = label_rate

        self.train_chunk_len = self.second2frame(train_chunk_len_s)
        self.train_chunk_hop_len = self.second2frame(train_chunk_hop_len_s)

        self.gt_chunk_len = int(train_chunk_len_s * self.label_rate) # label_rate * n_sec
        self.gt_chunk_hop_len = int(train_chunk_hop_len_s * self.label_rate)

        self.chunk_len = None
        self.chunk_hop_len = None
        self.feature_rate = self.fs / self.hop_len  # Frame rate per second
        self.label_upsample_ratio = int(self.feature_rate / self.label_rate)

        self.n_channels = n_channels
        self.n_bins = n_bins
        print("Feature shape : ({}, {}, {})".format(self.n_channels, self.train_chunk_len, self.n_bins))

    def second2frame(self, second: float = None) -> int:
        """
        Convert seconds to frame unit.
        """
        sample = int(second * self.fs)
        frame = int(round(sample/self.hop_len))
        return frame

    def get_segment_idxes(self, n_frames: int, downsample_ratio: int, pointer: int):
        # Get number of frame using segment rate
        assert n_frames % downsample_ratio == 0, 'n_features_frames is not divisible by downsample ratio'
        n_crop_frames = n_frames // downsample_ratio
        assert self.chunk_len // downsample_ratio <= n_crop_frames, 'Number of cropped frame is less than chunk len'

        idxes = np.arange(pointer,
                          pointer + n_crop_frames - self.chunk_len // downsample_ratio + 1,
                          self.chunk_hop_len // downsample_ratio).tolist()

        # Include the leftover of the cropped data
        if (n_crop_frames - self.chunk_len // downsample_ratio) % (self.chunk_hop_len // downsample_ratio) != 0:
            idxes.append(pointer + n_crop_frames - self.chunk_len // downsample_ratio)
        pointer += n_crop_frames

        return idxes, pointer


    def get_split(self, split : str = "train"):
        all_filenames = []
        feature_dir = os.path.join(self.feat_label_dir, split, "tracks")
        label_dir = os.path.join(self.feat_label_dir, split, "metadata")
        
        for root, dirnames, filenames in os.walk(feature_dir):
            for fname in filenames:
                if fname.endswith(".npy"):
                    all_filenames.append(os.path.join(root,fname))

        all_filenames = list(set(all_filenames))
        print("Total number of files : {}".format(len(all_filenames)))

        # Get chunk len and chunk hop len
        if split == "train":
            self.chunk_len = self.train_chunk_len
            self.chunk_hop_len = self.train_chunk_hop_len
        elif split == "test":
            self.chunk_len = self.train_chunk_len
            self.chunk_hop_len = self.train_chunk_len

        # Load and crop data
        features, labels, feature_chunk_idxes, gt_chunk_idxes, filename_list, test_batch_size = \
            self.load_chunk_data(split_filenames=all_filenames)

        # pack data
        db_data = {
            'features': features,
            'multi_accddoa_targets' : labels,
            'feature_chunk_idxes': feature_chunk_idxes,
            'gt_chunk_idxes': gt_chunk_idxes,
            'filename_list': filename_list,
            'test_batch_size': test_batch_size,
            'feature_chunk_len': self.chunk_len,
            'gt_chunk_len': self.chunk_len // self.label_upsample_ratio
        }

        print("Data loaded for {} split!".format(split))

        return db_data

    def load_chunk_data(self, split_filenames: List):
        feature_pointer = 0
        gt_pointer = 0
        features_list = []
        filename_list = []
        accdoa_target_list = []
        feature_idxes_list = []
        gt_idxes_list = []
        with Progress(transient=True) as progress:
            task = progress.add_task("[cyan]Loading files", total=len(split_filenames))

            for filename in split_filenames:
                # Load features -> n_channels x n_frames x n_features
                feature = np.load(filename)

                if feature.ndim == 2:
                    n_frames = feature.shape[0]
                    assert feature.shape[1] == int(self.n_channels * self.n_bins) , "Please check the feature space"
                    feature = feature.reshape(n_frames, self.n_channels, self.n_bins) # T, C, F
                    feature = feature.transpose((1,0,2)) # C, T, F

                # Load gt info from metadata
                accddoa = np.load(filename.replace("tracks", "metadata"))

                # We match the feature length with the number of ground truth labels that we have
                n_gt_frames = accddoa.shape[0]
                n_frames = n_gt_frames * 8
                if feature.shape[1] != n_frames : feature = feature[:, :n_frames]

                # Get sed segment indices
                feature_idxes, feature_pointer = self.get_segment_idxes(
                    n_frames=n_frames, downsample_ratio=1, pointer=feature_pointer)

                # Get gt segment indices
                gt_idxes, gt_pointer = self.get_segment_idxes(
                    n_frames=n_frames, downsample_ratio=self.label_upsample_ratio, pointer=gt_pointer) # Fixed at temporal downsample rate of 8x

                assert len(feature_idxes) == len(gt_idxes), 'nchunks for sed and gt are different'

                # Append data
                features_list.append(feature)
                filename_list.extend([filename] * len(feature_idxes))
                accdoa_target_list.append(accddoa)
                feature_idxes_list.append(feature_idxes)
                gt_idxes_list.append(gt_idxes)

                # Progress the progress bar
                progress.update(task, advance=1)

        if len(features_list) > 0:
            features = np.concatenate(features_list, axis=1)
            accddoa_targets = np.concatenate(accdoa_target_list, axis=0)
            feature_chunk_indexes = np.concatenate(feature_idxes_list, axis=0)
            gt_chunk_idxes = np.concatenate(gt_idxes_list, axis=0)
            test_batch_size = len(feature_idxes)  # to load all chunks of the same file
            return features, accddoa_targets, feature_chunk_indexes, gt_chunk_idxes, filename_list, test_batch_size
        else:
            return None, None, None, None, None, None

class seldDataset(Dataset):
    def __init__(self, db_data, transform=None):
        super().__init__()
        self.features = db_data['features']
        self.multi_accddoa_targets = db_data['multi_accddoa_targets']
        self.chunk_idxes = db_data['feature_chunk_idxes']
        self.gt_chunk_idxes = db_data['gt_chunk_idxes']
        self.filename_list = db_data['filename_list']
        self.chunk_len = db_data['feature_chunk_len']
        self.gt_chunk_len = db_data['gt_chunk_len']
        self.transform = transform  # transform that does not change label
        self.n_samples = len(self.chunk_idxes)
        
        print("seldDataset intiailized!\n\tNumber of batches : {}".format(self.n_samples))

    def __len__(self):
        """
        Total of training samples.
        """
        return self.n_samples

    def __getitem__(self, index):
        """
        Generate one sample of data
        """
        # Select sample
        chunk_idx = self.chunk_idxes[index]
        gt_chunk_idx = self.gt_chunk_idxes[index]

        # Load data and get label
        X = torch.tensor(self.features[:, chunk_idx: chunk_idx + self.chunk_len, :], dtype=torch.float32)
        
        # Mask the ACCDOA labels
        accdoa_targets = self.multi_accddoa_targets[gt_chunk_idx:gt_chunk_idx + self.gt_chunk_len]
        mask = accdoa_targets[:, :3] # SED one-hot encoding act as a mask
        mask = np.tile(mask, 2) # Create 2 (X, Y)
        accdoa_targets = mask * accdoa_targets[:, 3:] # Apply to the DOA labels
        target_labels = torch.tensor(accdoa_targets, dtype=torch.float32)

        if self.transform is not None:
            X = self.transform(X)

        return X, target_labels

if __name__ == '__main__':


    dataset = seldDatabase()
    # train_data = dataset.get_split("train")
    test_data = dataset.get_split("test")
    
    print("Test batch size : {}".format(test_data['test_batch_size']))
    
    # train_dataset = seldDataset(db_data=train_data)
    test_dataset = seldDataset(db_data=test_data)
    
    dataloader = DataLoader(test_dataset, batch_size=32, num_workers=0, pin_memory=True, shuffle=False)

    
    print('Number of batches: {}'.format(len(dataloader)))
    
    for train_iter, (X, targets) in enumerate(dataloader):
        print("Iteration {}/{}".format(train_iter+1, len(dataloader)), end='\r')
        if train_iter == 0:
            print('X: dtype: {} - shape: {}'.format(X.dtype, X.shape))
            print('ACCDOA: dtype: {} - shape: {}'.format(targets.dtype, targets.shape))
            print('ACCDOA sample: {}'.format(targets[-3]))
            break
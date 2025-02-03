import os 
import numpy as np 
import librosa
from sklearn import preprocessing


def extract_salsalite(audio_data, normalize=True):
    """
    Extract SALSA-Lite features from multi-channel audio.
    
    Parameters:
      audio_data : np.ndarray
          Input audio array of shape (n_mics, n_samples).
      normalize : bool
          If True, perform Z-score normalization of the log-power spectrogram
          per frequency bin (over mics and frames).
    
    Returns:
      audio_feature : np.ndarray
          Concatenated feature array of shape ((n_mics + n_mics-1), n_frames, n_cropped_bins)
          containing normalized log-power spectrograms (for each mic) and spatial phase features.
    """

    fs = 24000
    n_fft = 512
    hop_length = 300

    # DOA parameters
    n_mics = 4
    fmin_doa = 50
    fmax_doa = 4000  # initial upper bound for DOA
    d_max = 49 / 1000  # Maximum distance between two microphones (meters)
    f_alias = 343 / (2 * d_max)  # Spatial aliasing frequency
    fmax_doa = np.min((fmax_doa, fs // 2, f_alias))  
    
    n_bins = n_fft // 2 + 1
    lower_bin = int(np.floor(fmin_doa * n_fft / float(fs)))
    upper_bin = int(np.floor(fmax_doa * n_fft / float(fs)))
    lower_bin = np.max((1, lower_bin))

    # Cutoff frequency for spectrograms
    fmax = 9000  # Hz, meant to reduce feature dimensions
    cutoff_bin = int(np.floor(fmax * n_fft / float(fs)))  # 9000 Hz, 512 nfft: cutoff_bin = 192
    assert upper_bin <= cutoff_bin, 'Upper bin for spatial feature is higher than cutoff bin for spectrogram!'

    # Normalization factor for salsa_lite --> 2*pi*f/c
    c = 343
    delta = 2 * np.pi * fs / (n_fft * c)
    freq_vector = np.arange(n_bins)
    freq_vector[0] = 1
    freq_vector = freq_vector[:, None, None]  # n_bins x 1 x 1

    # Extract the features from the audio data
    log_specs = []

    for imic in np.arange(n_mics):
        audio_mic_data = audio_data[imic, :]
        stft = librosa.stft(y=np.asfortranarray(audio_mic_data), 
                            n_fft=n_fft, 
                            hop_length=hop_length,
                            center=True, 
                            window='hann', 
                            pad_mode='reflect')
        if imic == 0:
            n_frames = stft.shape[1]
            X = np.zeros((n_bins, n_frames, n_mics), dtype='complex')  # (n_bins, n_frames, n_mics)
        X[:, :, imic] = stft
        # Compute log linear power spectrum
        spec = (np.abs(stft) ** 2).T
        log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
        log_spec = np.expand_dims(log_spec, axis=0)
        log_specs.append(log_spec)
    log_specs = np.concatenate(log_specs, axis=0)  # (n_mics, n_frames, n_bins)

    # Normalize Log Power Spectra if Requested
    if normalize:
        # Compute mean and std over mics and frames (axes 0 and 1), leaving frequency dimension.
        mean = np.mean(log_specs, axis=(0, 1), keepdims=True)  # shape: (1, 1, n_bins)
        std = np.std(log_specs, axis=(0, 1), keepdims=True)    # shape: (1, 1, n_bins)
        std[std == 0] = 1  # avoid division by zero
        log_specs = (log_specs - mean) / std

    # Compute spatial feature
    phase_vector = np.angle(X[:, :, 1:] * np.conj(X[:, :, 0, None]))
    phase_vector = phase_vector / (delta * freq_vector)
    phase_vector = np.transpose(phase_vector, (2, 1, 0))  # (n_mics, n_frames, n_bins)

    # Crop frequency
    log_specs = log_specs[:, :, lower_bin:cutoff_bin]
    phase_vector = phase_vector[:, :, lower_bin:cutoff_bin]
    phase_vector[:, :, upper_bin:] = 0

    # Stack features
    audio_feature = np.concatenate((log_specs, phase_vector), axis=0)

    return audio_feature

def load_output_format_file(file):
    _output_dict = {}
    _fid = open(file, 'r')
    _words = []     # For empty files
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        _output_dict[_frame_ind].append([int(_words[1]), float(_words[2])]) # Class Index, Azimuth Angle
    _fid.close()
    
    return _output_dict

def convert_output_format_polar_to_cartesian(in_dict):
    out_dict = {}
    for frame_ind, events in in_dict.items():
        out_dict[frame_ind] = []
        for event in events:
            # Convert azimuth from degrees to radians
            azi_rad = event[1] * np.pi / 180.0
            x = np.cos(azi_rad)
            y = np.sin(azi_rad)
            out_dict[frame_ind].append([event[0], x, y])
    return out_dict

def get_labels_for_file(_desc_file, _nb_label_frames, _nb_unique_classes=3):
    """
    Constructs the label matrix for a file given the description dictionary.
    
    The label matrix has shape (nb_label_frames, 3*nb_unique_classes), where the columns
    correspond to [SED, x, y] for each class.
    """

    se_label = np.zeros((_nb_label_frames, _nb_unique_classes))
    x_label = np.zeros((_nb_label_frames, _nb_unique_classes))
    y_label = np.zeros((_nb_label_frames, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            for active_event in active_event_list:
                #print(active_event)
                se_label[frame_ind, active_event[0]] = 1
                x_label[frame_ind, active_event[0]] = active_event[1]
                y_label[frame_ind, active_event[0]] = active_event[2]

    label_mat = np.concatenate((se_label, x_label, y_label), axis=1)
    return label_mat

if __name__ == "__main__":

    output_dir = "./output_data_block_demo5"
    feat_dir = output_dir.replace("output_data", "feat_label")
    # feat_dir = "./feat_label_block_silence"
    fs = 24000
    label_rate = 10

    for root, dirnames, filenames in os.walk(output_dir, topdown=True):
        for filename in filenames:
            if filename.endswith(".wav"):

                # Load the audio data
                filepath = os.path.join(root, filename)
                audio_data , _ = librosa.load(filepath, sr=fs, mono=False, dtype=np.float32)

                # Determine the number of feature frames
                nb_feat_frames = int(audio_data.shape[1] / 300.0)
                nb_label_frames = int(audio_data.shape[1]/ fs * label_rate) # 10 fps

                # Extract the SALSA-Lite features
                _feat = extract_salsalite(audio_data=audio_data)
                _feat = _feat[:, :nb_feat_frames, :]

                # Create the new directory for the features
                new_filepath = filepath.replace("output_data", "feat_label")
                new_filedir = os.path.dirname(new_filepath)
                os.makedirs(new_filedir, exist_ok=True)

                new_feat_fname = new_filepath.replace(".wav", ".npy")
                np.save(new_feat_fname, _feat)

                # Now, we extract the labels
                metadata_file = filepath.replace("tracks", "metadata").replace(".wav", ".csv")
                desc_file_polar = load_output_format_file(metadata_file)
                desc_file = convert_output_format_polar_to_cartesian(desc_file_polar)
                accdoa_labels = get_labels_for_file(desc_file, nb_label_frames)

                # Create the new directory for the metadata labels
                new_metadata_filepath = metadata_file.replace("output_data", "feat_label")
                os.makedirs(os.path.dirname(new_metadata_filepath), exist_ok=True)

                new_label_fname = new_metadata_filepath.replace(".csv", ".npy")
                np.save(new_label_fname, accdoa_labels)

                # Verbose printing
                print("{}: {}, {}".format(new_feat_fname, _feat.shape, accdoa_labels.shape))

    # # Initialize spec_scalers as None; it will be initialized after loading the first file
    # spec_scalers = None
    # for root, dirs, filenames in os.walk(feat_dir):
    #     for fname in filenames:
    #         fpath = os.path.join(root, fname)
    #         if "tracks" in fpath:
    #             print("Normalizing for: {}".format(fpath))
    #             feat_file = np.load(fpath)
    #             if spec_scalers is None:
    #                 num_channels = feat_file.shape[0]
    #                 spec_scalers = [preprocessing.StandardScaler() for _ in range(num_channels)]
    #                 print(f'Initialized {num_channels} StandardScalers for each channel.')

    #             # Fit each scaler with the data from its respective channel
    #             for ch in range(num_channels):
    #                 channel_data = feat_file[ch].reshape(-1, feat_file.shape[2])  # Shape: (time, frequency)
    #                 spec_scalers[ch].partial_fit(channel_data)

    #             # Clean up
    #             del feat_file

    # print('Normalizing feature files...')
    # for root, dirs, filenames in os.walk(feat_dir):
    #     for fname in filenames:
    #         fpath = os.path.join(root, fname)
    #         if "tracks" in fpath:
    #             feat_file = np.load(fpath)

    #             # Apply the scaler for each channel
    #             for ch in range(num_channels):
    #                 # Reshape the data to (samples, features) if necessary
    #                 channel_data = feat_file[ch].reshape(-1, feat_file.shape[2])  # Shape: (time, frequency)
    #                 normalized_channel = spec_scalers[ch].transform(channel_data)
    #                 feat_file[ch] = normalized_channel.reshape(feat_file.shape[1], feat_file.shape[2])

    #             np.save(fpath, feat_file)
    #             print("Normalized for: {}, Shape: {}".format(fpath, feat_file.shape))
    #             del feat_file

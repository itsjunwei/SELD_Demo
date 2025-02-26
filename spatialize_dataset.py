import os
import numpy as np
import soundfile as sf
import random
import csv
from scipy.signal import convolve
import librosa

##############################################################################
# Helper Functions
##############################################################################

def measure_rms_multichannel(signal_4ch: np.ndarray) -> float:
    """
    Measure RMS of a multi-channel signal by flattening across channels.
    Expected shape: (num_samples, num_channels).
    """
    if signal_4ch.ndim != 2:
        raise ValueError("Expected a 2D array for multi-channel audio.")
    # Ensure shape = (num_samples, num_channels)
    if signal_4ch.shape[0] < signal_4ch.shape[1]:
        signal_4ch = signal_4ch.T
    flattened = signal_4ch.reshape(-1)
    return np.sqrt(np.mean(flattened**2))


def extract_random_segment(audio: np.ndarray,
                           sr: int,
                           segment_length: float) -> np.ndarray:
    """
    Extract a random segment (in seconds) from a longer 1D audio array.
    
    Parameters
    ----------
    audio : np.ndarray, shape (num_samples,)
        Single-channel audio.
    sr : int
        Sample rate.
    segment_length : float
        Desired segment length in seconds.

    Returns
    -------
    segment : np.ndarray
        1D array of shape (segment_samples,).
    """
    total_duration = len(audio) / sr
    if segment_length > total_duration:
        raise ValueError(f"Audio is shorter ({total_duration:.2f}s) than requested segment ({segment_length:.2f}s).")
    
    max_start_time = total_duration - segment_length
    start_time = random.uniform(0, max_start_time)
    start_sample = int(round(start_time * sr))
    end_sample = start_sample + int(round(segment_length * sr))
    return audio[start_sample:end_sample]


def extract_random_segment_4ch(audio_4ch: np.ndarray,
                               sr: int,
                               segment_length: float = 60.0) -> np.ndarray:
    """
    Extract a random 4-channel segment (in seconds) from a longer 4-channel array.
    Shape: (num_samples, 4).
    """
    total_duration = audio_4ch.shape[0] / sr
    if segment_length > total_duration:
        raise ValueError(f"4-channel audio is shorter ({total_duration:.2f}s) than requested segment.")
    
    max_start_time = total_duration - segment_length
    start_time = random.uniform(0, max_start_time)
    start_sample = int(round(start_time * sr))
    end_sample = start_sample + int(round(segment_length * sr))
    return audio_4ch[start_sample:end_sample, :]


def convolve_mono_with_4ch_rir(event_mono: np.ndarray,
                               rir_4ch: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-channel event with a 4-channel SRIR to produce a 4-channel event.
    """
    if rir_4ch.ndim != 2 or rir_4ch.shape[1] != 4:
        raise ValueError("rir_4ch must have shape (rir_samples, 4).")

    out_channels = []
    for c in range(4):
        ch_ir = rir_4ch[:, c]
        convolved = convolve(event_mono, ch_ir, mode='same')
        out_channels.append(convolved)
    # shape: (n_samples, 4)
    return np.column_stack(out_channels)


def mix_event_into_background_4ch(
    background_4ch: np.ndarray,
    event_4ch: np.ndarray,
    sr: int,
    start_time: float
):
    """
    Add the 4-channel event into the 4-channel background in-place.
    """
    start_sample = int(round(start_time * sr))
    end_sample = start_sample + event_4ch.shape[0]
    bg_len = background_4ch.shape[0]

    if start_sample >= bg_len:
        return
    if end_sample > bg_len:
        overlap_len = bg_len - start_sample
        background_4ch[start_sample:start_sample+overlap_len, :] += event_4ch[:overlap_len, :]
    else:
        background_4ch[start_sample:end_sample, :] += event_4ch


##############################################################################
# New Class-Constrained Timeline Placement
##############################################################################

def place_events_class_based(
    event_durations: list[float],
    event_classes: list[int],
    total_duration: float = 60.0,
    time_resolution: float = 0.1,
    max_attempts_per_event: int = 1000
) -> list[float]:
    """
    Place events on a 100ms timeline (time_resolution=0.1), ensuring:
      - No more than 2 events overlap at any time.
      - Overlapping events must be from different classes
        (i.e., never allow two events of the same class in the same frame).

    This function returns a list of start times (seconds), or None if an event
    could not be placed without violating constraints.

    Parameters
    ----------
    event_durations : list[float]
        Duration in seconds for each event i.
    event_classes : list[int]
        Class label for each event i (0=alarm,1=impact,2=speech, etc.).
        Must have the same length as event_durations.
    total_duration : float
        The overall timeline length in seconds.
    time_resolution : float
        We quantize starts to multiples of this step (default 0.1 = 100ms).
    max_attempts_per_event : int
        How many times to attempt random placement for each event.

    Returns
    -------
    start_times : list[float or None]
        The chosen start time in seconds for each event, or None if not placed.
    """
    total_frames = int(np.floor(total_duration / time_resolution))
    # We'll store a list of (class_label, start_frame, end_frame) for already placed events
    placed_intervals = []
    start_times = []

    for dur, cls in zip(event_durations, event_classes):
        placed = False
        attempts = 0
        event_frames = int(np.ceil(dur / time_resolution))

        # If event is longer than the entire timeline, skip
        if event_frames > total_frames:
            start_times.append(None)
            continue

        # Attempt random placements
        while not placed and attempts < max_attempts_per_event:
            attempts += 1
            start_frame = random.randint(0, total_frames - event_frames)
            candidate_interval = (start_frame, start_frame + event_frames)

            # Collect classes that overlap with this candidate
            overlapping_classes = set()
            # Also check how many events overlap in total
            overlap_count = 0

            for (c_label, s_frame, e_frame) in placed_intervals:
                # If intervals overlap
                if not (candidate_interval[1] <= s_frame or candidate_interval[0] >= e_frame):
                    overlap_count += 1
                    overlapping_classes.add(c_label)

            # Now check constraints:
            # 1) No more than 2 total events can overlap
            if overlap_count >= 2:
                continue

            # 2) We cannot overlap with same class
            if cls in overlapping_classes:
                continue

            # 3) If there's 1 overlapping event so far (different class),
            #    that's fine. That makes 2 total classes.
            #    If there's 0 overlapping, also fine.

            # Because overlap_count < 2 and we haven't found 'cls' in overlapping_classes,
            # we can place the event
            placed_intervals.append((cls, candidate_interval[0], candidate_interval[1]))
            # Convert frames -> seconds
            start_sec = start_frame * time_resolution
            start_times.append(start_sec)
            placed = True

        if not placed:
            start_times.append(None)
            print(f"Warning: Could not place event of duration {dur:.2f}s (class={cls}) "
                  f"without exceeding constraints. Skipping.")

    return start_times


##############################################################################
# Main Synthesis Function
##############################################################################

def create_spatialized_mix_from_class_audio(
    ambience_path_4ch: str,
    class_audio_paths_mono: dict[int, str],
    srir_folder_4ch: str,
    out_audio_path_4ch: str,
    out_csv_path: str,
    sr: int = 24000,
    segment_length: float = 60.0,
    num_events: int = 5,
    snr_range_db: tuple[float, float] = (0, 10),
    max_polyphony: int = 2,
    time_resolution: float = 0.1,
    possible_angles: list[int] = [0, 20, 40, 60, 80, 100, 260, 280, 300, 320, 340],
    min_event_length: float = 0.5,
    max_event_length: float = 3.0,
    use_500ms_blocks: bool = False,
    block_duration: float = 0.5,
):
    """
    Create a 4-channel synthetic mixture from:
      1. A 4-channel ambience (segment_length e.g. 60s),
      2. Several monophonic event snippets. The class->label mapping is:
            0 -> Alarm, 1 -> Impact, 2 -> Speech, etc.
      3. Convolution with a random 4-channel SRIR from chosen angle,
      4. If use_500ms_blocks=False (default):
           - Place events on a 100 ms grid, random snippet lengths in [min_event_length, max_event_length].
           - Up to max_polyphony=2 can overlap. Standard approach.
         If use_500ms_blocks=True:
           - Place each event in multiples of block_duration=0.5s.  The event can span N consecutive blocks
             (N chosen by rounding up from [min_event_length, max_event_length]).
           - Up to 2 events can occupy the same 500ms block at once.
      5. Scale to random SNR,
      6. Output final 4-channel .wav,
      7. Output CSV ground truth with 100ms frames (time_resolution=0.1).

    Parameters
    ----------
    ambience_path_4ch : str
        Path to the 4-channel ambience file (wav).
    class_audio_paths_mono : dict[int, str]
        e.g. {0:"alarm_concat.wav", 1:"impact_concat.wav", 2:"speech_concat.wav"}
    srir_folder_4ch : str
        Folder containing 4-channel SRIRs named such that angles can be matched.
    out_audio_path_4ch : str
        Where to write the final 4-ch mixture wav.
    out_csv_path : str
        Where to write the CSV ground truth.
    sr : int
        Sample rate (assumed consistent).
    segment_length : float
        E.g. 60 for 1 minute.
    num_events : int
        How many total events to overlay if use_500ms_blocks=False.
        (When use_500ms_blocks=True, we still "attempt" to place these many events.)
    snr_range_db : tuple[float, float]
        Range for random SNR selection in dB.
    max_polyphony : int
        Up to 2 overlapping events.
    time_resolution : float
        0.1 => final labeling frames are each 100ms.
    possible_angles : list[int]
        Angles for which SRIRs exist.
    min_event_length, max_event_length : float
        (seconds) The random event durations used in either snippet or block approach.
    use_500ms_blocks : bool
        Toggle for new block-based approach. If True, each event spans N multiples of block_duration.
    block_duration : float
        Size of each block in seconds (e.g. 0.5). Only used when use_500ms_blocks=True.

    Returns
    -------
    None (writes mixture wav + CSV to disk).
    """

    # --------------------------
    # 1) Load the 4-ch ambience
    # --------------------------
    import librosa
    ambience_4ch, sr_amb = librosa.load(ambience_path_4ch, sr=sr, mono=False, dtype=np.float32)
    ambience_4ch = ambience_4ch.T
    if ambience_4ch.ndim != 2 or ambience_4ch.shape[1] != 4:
        raise ValueError("Ambience must be 4-channel (samples,4).")

    ambience_segment_4ch = extract_random_segment_4ch(ambience_4ch, sr, segment_length)
    final_mix_4ch = np.copy(ambience_segment_4ch)

    # For labeling
    total_frames = int(np.floor(segment_length / time_resolution))

    # --------------------------
    # 2) Pre-load each class's audio
    # --------------------------
    class_audio_data = {}
    for cls, mono_path in class_audio_paths_mono.items():
        audio_mono, sr_cls = sf.read(mono_path)
        if audio_mono.ndim != 1:
            raise ValueError(f"Class file {mono_path} must be mono.")
        class_audio_data[cls] = audio_mono

    # We'll store labeling as a list of (frame_idx, class_label, angle)
    frame_label_records = []

    # -------------------------------------------------------------------------
    # MODE A) use_500ms_blocks = False: Original snippet-based approach
    # -------------------------------------------------------------------------
    if not use_500ms_blocks:
        # i) Pick random snippet durations in [min_event_length, max_event_length]
        event_snippets = []
        event_classes = []
        event_durations = []
        class_labels = list(class_audio_paths_mono.keys())

        for _ in range(num_events):
            chosen_class = random.choice(class_labels)
            length_sec = random.uniform(min_event_length, max_event_length)
            snippet = extract_random_segment(class_audio_data[chosen_class], sr, length_sec)
            event_snippets.append(snippet)
            event_classes.append(chosen_class)
            event_durations.append(len(snippet)/sr)

        # ii) Place events with place_events_class_based
        start_times = place_events_class_based(
            event_durations=event_durations,
            event_classes=event_classes,
            total_duration=segment_length,
            time_resolution=time_resolution
        )

        # measure ambience RMS
        ambience_rms = measure_rms_multichannel(final_mix_4ch)

        # iii) For each event, convolve & mix
        for i, (snippet_mono, dur_sec) in enumerate(zip(event_snippets, event_durations)):
            start_sec = start_times[i]
            if start_sec is None:
                continue
            class_label = event_classes[i]
            angle_chosen = random.choice(possible_angles)

            # Find SRIR
            srir_candidates = [
                f for f in os.listdir(srir_folder_4ch)
                if f == f"{angle_chosen}.wav"
            ]
            
            assert len(srir_candidates) == 1, "More than 1 SRIR candidate found -- {}".format(len(srir_candidates))
            
            if not srir_candidates:
                print(f"No SRIR found for angle={angle_chosen}, skipping.")
                continue

            srir_file = random.choice(srir_candidates)
            srir_path = os.path.join(srir_folder_4ch, srir_file)
            srir_4ch, sr_srir = sf.read(srir_path)
            if srir_4ch.ndim != 2 or srir_4ch.shape[1] != 4:
                print(f"SRIR {srir_file} is not 4-ch. Skipping.")
                continue

            # Convolve
            event_4ch = convolve_mono_with_4ch_rir(snippet_mono, srir_4ch)
            event_rms = measure_rms_multichannel(event_4ch)
            if event_rms == 0:
                continue

            snr_db = random.uniform(*snr_range_db)
            desired_event_rms = ambience_rms * (10.0 ** (snr_db / 20.0))
            scale_factor = desired_event_rms / event_rms
            event_4ch_scaled = event_4ch * scale_factor

            mix_event_into_background_4ch(final_mix_4ch, event_4ch_scaled, sr, start_sec)

            # Label frames at 100ms
            end_sec = start_sec + (len(event_4ch_scaled)/sr)
            frame_start = int(np.floor(start_sec / time_resolution))
            frame_end = int(np.ceil(end_sec / time_resolution))
            frame_start = max(frame_start, 0)
            frame_end = min(frame_end, total_frames)
            for f_idx in range(frame_start, frame_end):
                frame_label_records.append((f_idx, class_label, angle_chosen))

    # -------------------------------------------------------------------------
    # MODE B) use_500ms_blocks = True: Block-based approach with N-block events
    # -------------------------------------------------------------------------
    else:
        block_count = int(segment_length // 0.5)  # e.g. 120 blocks if 60s
        frames_per_block = int(0.5 / time_resolution)  # e.g. 5 frames for each 0.5s block
        block_class_sets = [set() for _ in range(block_count)]  # track which classes occupy each block
        ambience_rms = measure_rms_multichannel(final_mix_4ch)
        class_labels = list(class_audio_paths_mono.keys())

        # We'll store (start_block, n_blocks, class_label) for each event
        scheduled_events = []

        for _ in range(num_events):
            chosen_class = random.choice(class_labels)
            rand_len = random.uniform(min_event_length, max_event_length)
            n_blocks = int(np.ceil(rand_len / 0.5))
            if n_blocks <= 0 or n_blocks > block_count:
                continue

            # attempt random block placement
            placed = False
            max_attempts = 2000
            attempt_cnt = 0
            while not placed and attempt_cnt < max_attempts:
                attempt_cnt += 1
                start_b = random.randint(0, block_count - n_blocks)
                # check each block in [start_b, start_b+n_blocks)
                can_place = True
                for b_i in range(start_b, start_b + n_blocks):
                    if len(block_class_sets[b_i]) >= max_polyphony:
                        can_place = False
                        break
                    # if you also want to forbid 2 events of same class:
                    if chosen_class in block_class_sets[b_i]:
                        can_place = False
                        break

                if can_place:
                    # place
                    for b_i in range(start_b, start_b + n_blocks):
                        block_class_sets[b_i].add(chosen_class)
                    scheduled_events.append((start_b, n_blocks, chosen_class))
                    placed = True

        # Now we convolve/mix those scheduled events
        for (start_b, nb, cls_label) in scheduled_events:
            angle_chosen = random.choice(possible_angles)
            srir_candidates = [
                f for f in os.listdir(srir_folder_4ch)
                if f.endswith(".wav") and f == f"{angle_chosen}.wav"
            ]
            if not srir_candidates:
                print(f"No SRIR for angle {angle_chosen}, skip.")
                continue
            srir_file = random.choice(srir_candidates)
            srir_path = os.path.join(srir_folder_4ch, srir_file)
            srir_4ch, sr_srir = sf.read(srir_path)
            if srir_4ch.ndim != 2 or srir_4ch.shape[1] != 4:
                print(f"SRIR {srir_file} invalid shape. Skipping.")
                continue

            event_len_s = nb * 0.5
            snippet_mono = extract_random_segment(class_audio_data[cls_label], sr, event_len_s)
            if len(snippet_mono) == 0:
                continue

            # Convolution
            event_4ch = convolve_mono_with_4ch_rir(snippet_mono, srir_4ch)
            event_rms = measure_rms_multichannel(event_4ch)
            if event_rms == 0:
                continue

            # SNR scaling
            snr_db = random.uniform(*snr_range_db)
            desired_event_rms = ambience_rms * (10.0 ** (snr_db / 20.0))
            scale_factor = desired_event_rms / event_rms
            event_4ch_scaled = event_4ch * scale_factor

            # Mixing
            start_time_s = start_b * 0.5
            mix_event_into_background_4ch(final_mix_4ch, event_4ch_scaled, sr, start_time_s)

            # labeling
            block_frame_start = start_b * frames_per_block
            block_frame_end = block_frame_start + (nb * frames_per_block)
            block_frame_end = min(block_frame_end, total_frames)
            for f_idx in range(block_frame_start, block_frame_end):
                frame_label_records.append((f_idx, cls_label, angle_chosen))

    # ------------------------------
    # 7) Write final mixture
    # ------------------------------

    # Prevent clipping
    peak = np.max(np.abs(final_mix_4ch))
    final_mix_4ch = final_mix_4ch / peak

    assert sr_amb == sr_cls == sr_srir, "Sampling rates are off : {}/{}/{}".format(sr_amb, sr_cls, sr_srir)

    sf.write(out_audio_path_4ch, final_mix_4ch, sr)
    # print(f"Created 4-channel mixture: {out_audio_path_4ch}")

    # ------------------------------
    # 8) Write CSV
    # ------------------------------
    frame_label_records.sort(key=lambda x: x[0])
    with open(out_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for (frame_idx, label, angle) in frame_label_records:
            writer.writerow([frame_idx, label, angle])

    # print(f"Ground truth CSV saved to: {out_csv_path}")


##############################################################################
# Feature Extraction Functions
##############################################################################

def extract_salsalite(audio_data, normalize=True):

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

    # Normalize Log Power Spectra
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
                se_label[frame_ind, active_event[0]] = 1
                x_label[frame_ind, active_event[0]] = active_event[1]
                y_label[frame_ind, active_event[0]] = active_event[2]

    label_mat = np.concatenate((se_label, x_label, y_label), axis=1)
    return label_mat


if __name__ == "__main__":
    from rich.progress import Progress

    output_dir = "./output_data_2fps_5sec_dr2"
    os.makedirs(output_dir, exist_ok=True)
    rooms = os.listdir("./normalized_rirs")
    if "2fps" in output_dir:
        label_rate = 2
    else:
        label_rate = 10

    splits = ["train", "test"]
    rooms = ['DemoRoom_A']

    for split in splits:
        if split == "train":
            n_tracks = 720 * 6 # 12 hours
        elif split == "test":
            n_tracks = 720 * 2 # 2 hours

        ambience_files = [os.path.join(f"./ambience/{split}", d) for d in os.listdir(f"./ambience/{split}")]

        class_audio_dict = {
            0: f"./cleaned_concat_audio/alarm_{split}.wav",
            1: f"./cleaned_concat_audio/impact_{split}.wav",
            2: f"./cleaned_concat_audio/speech_{split}.wav"
        }

        for room in rooms:
            srir_folder = os.path.join("./normalized_rirs", room)

            track_dir = os.path.join(output_dir, split, "tracks", room)
            os.makedirs(track_dir, exist_ok=True)

            csv_dir = os.path.join(output_dir, split, "metadata", room)
            os.makedirs(csv_dir, exist_ok=True)
            
            with Progress() as progress:
                task = progress.add_task("[green]Mixing tracks for {}: ".format(split), total=n_tracks)

                for ith_track in range(n_tracks):
                    ambience_file_4ch = random.choice(ambience_files)

                    output_4ch_wav = os.path.join(track_dir, "track_{}.wav".format(ith_track+1))
                    output_csv = os.path.join(csv_dir, "track_{}.csv".format(ith_track+1))

                    create_spatialized_mix_from_class_audio(
                        ambience_path_4ch=ambience_file_4ch,
                        class_audio_paths_mono=class_audio_dict,
                        srir_folder_4ch=srir_folder,
                        out_audio_path_4ch=output_4ch_wav,
                        out_csv_path=output_csv,
                        sr=24000,
                        segment_length=5.0,     # Duration of each clip
                        num_events=2,
                        snr_range_db=(0, 35),
                        max_polyphony=2,
                        time_resolution=0.5,  # 100 ms frames
                        possible_angles=[20, 60, 300, 340],
                        min_event_length=1.0,
                        max_event_length=3.0,
                        use_500ms_blocks=True
                    )
                    progress.update(task, advance=1)

    # Feature Extraction for the Spatialized Dataset
    feat_dir = output_dir.replace("output_data", "feat_label")
    fs = 24000

    for root, dirnames, filenames in os.walk(output_dir, topdown=True):
        with Progress() as progress:
            task = progress.add_task("[red]Generating features: ", total=len(filenames))
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
                    progress.console.log(f"{new_feat_fname}: {_feat.shape}, {accdoa_labels.shape}")
                    progress.update(task, advance=1)
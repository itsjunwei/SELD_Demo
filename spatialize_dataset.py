import os
import numpy as np
import soundfile as sf
import random
import csv
from scipy.signal import convolve

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

def place_events_on_timeline(
    event_durations: list[float],
    total_duration: float = 60.0,
    max_polyphony: int = 2,
    time_resolution: float = 0.1,
    max_attempts_per_event: int = 1000
) -> list[float]:
    """
    Place each event on a timeline quantized to `time_resolution` with up to `max_polyphony`.
    Returns a list of start times (seconds). `None` if placement fails.
    """
    total_frames = int(np.floor(total_duration / time_resolution))
    placed_intervals = []
    start_times = []

    for dur in event_durations:
        placed = False
        attempt_count = 0
        event_frames = int(np.ceil(dur / time_resolution))

        if event_frames > total_frames:
            # can't place an event longer than the entire timeline
            start_times.append(None)
            continue

        while not placed and attempt_count < max_attempts_per_event:
            attempt_count += 1
            start_frame = random.randint(0, total_frames - event_frames)
            candidate_interval = (start_frame, start_frame + event_frames)

            overlap_count = 0
            for (s_frame, e_frame) in placed_intervals:
                # Overlap check
                if not (candidate_interval[1] <= s_frame or candidate_interval[0] >= e_frame):
                    overlap_count += 1
                    if overlap_count >= max_polyphony:
                        break
            
            if overlap_count < max_polyphony:
                placed_intervals.append(candidate_interval)
                start_times.append(start_frame * time_resolution)
                placed = True
        
        if not placed:
            start_times.append(None)
            print(f"Warning: Could not place event of duration {dur:.2f}s with polyphony={max_polyphony}. Skipping.")
    
    return start_times

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
        convolved = convolve(event_mono, ch_ir, mode='full')
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
    max_event_length: float = 3.0
):
    """
    Create a 4-channel synthetic mixture from:
      1. A 4-channel ambience (1-minute segment),
      2. Several monophonic event snippets. Each snippet is extracted from one of
         the big class-concatenated audio files. The class -> numeric label is:
            0 -> Alarm
            1 -> Impact
            2 -> Speech
      3. Each snippet is convolved with a random 4-channel SRIR from a chosen angle,
      4. Placed on a 100 ms grid (time_resolution=0.1) with max polyphony=2,
      5. Scaled to random SNR,
      6. Output: final 4-channel .wav + CSV with frame-level labels:
         (frame_index, numeric_label, angle).

    Parameters
    ----------
    ambience_path_4ch : str
        Path to 4-channel ambience file.
    class_audio_paths_mono : dict of {int: str}
        e.g. {0: "alarm_concat.wav", 1: "impact_concat.wav", 2: "speech_concat.wav"}
    srir_folder_4ch : str
        Folder with 4-channel SRIR files for discrete angles.
    out_audio_path_4ch : str
        Output path for final 4-channel mixture.
    out_csv_path : str
        Output path for CSV ground truth.
    sr : int
        Sample rate.
    segment_length : float
        How many seconds of ambience to extract (default=60).
    num_events : int
        Total number of events to overlay.
    snr_range_db : tuple of float
        Range of SNR in dB (min, max).
    max_polyphony : int
        Maximum overlap of events.
    time_resolution : float
        Step size in seconds (default=0.1 => 100 ms).
    possible_angles : list[int]
        Discrete angles for which we have SRIRs.
    min_event_length : float
        Min random length (in seconds) for an event snippet.
    max_event_length : float
        Max random length (in seconds) for an event snippet.
    """

    # -------------------------------------------------------------------------
    # 1) Load 4-ch Ambience and extract random 60 s
    # -------------------------------------------------------------------------
    ambience_4ch, sr_amb = sf.read(ambience_path_4ch)
    if ambience_4ch.ndim != 2 or ambience_4ch.shape[1] != 4:
        raise ValueError("Ambience must be 4-channel: shape (samples,4).")
    if sr_amb != sr:
        # Resample if needed
        pass

    ambience_segment_4ch = extract_random_segment_4ch(
        ambience_4ch, sr, segment_length
    )

    # -------------------------------------------------------------------------
    # 2) Load the large audio for each class (mono)
    # -------------------------------------------------------------------------
    class_audio_data = {}
    for class_label, path in class_audio_paths_mono.items():
        audio_mono, sr_class = sf.read(path)
        if audio_mono.ndim != 1:
            raise ValueError(f"Class file {path} must be mono.")
        if sr_class != sr:
            # Resample if needed
            pass
        class_audio_data[class_label] = audio_mono

    # We'll pick random event snippets from these class audios
    event_snippets = []
    event_classes = []
    event_durations = []

    # -------------------------------------------------------------------------
    # 3) Randomly pick events from the 3 classes
    # -------------------------------------------------------------------------
    class_labels = list(class_audio_paths_mono.keys())  # e.g. [0,1,2]
    for _ in range(num_events):
        # pick random class
        chosen_class = random.choice(class_labels)

        # pick random snippet length
        length_sec = random.uniform(min_event_length, max_event_length)
        # extract from that class's big file
        snippet = extract_random_segment(class_audio_data[chosen_class], sr, length_sec)

        event_snippets.append(snippet)
        event_classes.append(chosen_class)
        event_durations.append(len(snippet) / sr)

    # -------------------------------------------------------------------------
    # 4) Place events on timeline with max_polyphony=2, 100ms resolution
    # -------------------------------------------------------------------------
    start_times = place_events_class_based(
        event_durations=event_durations,
        event_classes=event_classes,
        total_duration=segment_length,
        time_resolution=time_resolution
    )

    # -------------------------------------------------------------------------
    # 5) Compute 4-ch Ambience RMS
    # -------------------------------------------------------------------------
    ambience_rms = measure_rms_multichannel(ambience_segment_4ch)
    final_mix_4ch = np.copy(ambience_segment_4ch)

    # For CSV labeling: list of (frame_idx, class_label, angle)
    frame_label_records = []
    total_frames = int(np.floor(segment_length / time_resolution))

    # -------------------------------------------------------------------------
    # 6) For each event, pick angle, SRIR, convolve, SNR, mix
    # -------------------------------------------------------------------------
    for i, (snippet_mono, dur_sec) in enumerate(zip(event_snippets, event_durations)):
        start_sec = start_times[i]
        if start_sec is None:
            # event wasn't placed
            continue

        class_label = event_classes[i]

        # 6a) pick a random angle
        angle_chosen = random.choice(possible_angles)

        # 6b) find SRIR file
        srir_candidates = [
            f for f in os.listdir(srir_folder_4ch)
            if f.endswith(".wav") and str(angle_chosen) in f  # naive pattern match
        ]
        if not srir_candidates:
            print(f"No SRIR found for angle={angle_chosen}, skipping event.")
            continue
        srir_file = random.choice(srir_candidates)
        srir_path = os.path.join(srir_folder_4ch, srir_file)

        srir_4ch, sr_srir = sf.read(srir_path)
        if srir_4ch.ndim != 2 or srir_4ch.shape[1] != 4:
            print(f"SRIR {srir_file} is not 4-ch. Skipping.")
            continue
        if sr_srir != sr:
            # Resample if needed
            pass

        # 6c) Convolve
        event_4ch = convolve_mono_with_4ch_rir(snippet_mono, srir_4ch)

        # 6d) SNR scaling
        event_rms = measure_rms_multichannel(event_4ch)
        if event_rms == 0:
            continue

        snr_db = random.uniform(*snr_range_db)
        desired_event_rms = ambience_rms * (10.0 ** (snr_db / 20.0))
        scale_factor = desired_event_rms / event_rms

        event_4ch_scaled = event_4ch * scale_factor

        # 6e) Mix
        mix_event_into_background_4ch(final_mix_4ch, event_4ch_scaled, sr, start_sec)

        # 6f) Ground-truth labeling, 100ms frames
        event_duration_out = event_4ch_scaled.shape[0] / sr
        end_sec = start_sec + event_duration_out

        frame_start = int(np.floor(start_sec / time_resolution))
        frame_end = int(np.ceil(end_sec / time_resolution))
        frame_start = max(0, frame_start)
        frame_end = min(total_frames, frame_end)

        # record: (frame_idx, numeric_label, angle_chosen)
        for f_idx in range(frame_start, frame_end):
            frame_label_records.append((f_idx, class_label, angle_chosen))

    # -------------------------------------------------------------------------
    # 7) Write final 4-channel mixture
    # -------------------------------------------------------------------------
    sf.write(out_audio_path_4ch, final_mix_4ch, sr)
    print(f"Created 4-channel mixture: {out_audio_path_4ch}")

    # -------------------------------------------------------------------------
    # 8) Write CSV ground truth
    # -------------------------------------------------------------------------
    # Format: frame_index, class_label, angle
    frame_label_records.sort(key=lambda x: x[0])
    with open(out_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(["frame_index", "class_label", "angle"])
        for (frame_idx, label, angle) in frame_label_records:
            writer.writerow([frame_idx, label, angle])

    print(f"Ground truth CSV saved to: {out_csv_path}")


if __name__ == "__main__":

    output_dir = "./output_data"
    os.makedirs(output_dir, exist_ok=True)
    rooms = os.listdir("./normalized_rirs")

    splits = ["train", "test"]

    for split in splits:
        if split == "train":
            n_tracks = 60
        elif split == "test":
            n_tracks = 10

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
                    segment_length=60.0,   # 1 minute
                    num_events=12,
                    snr_range_db=(10, 30),
                    max_polyphony=2,
                    time_resolution=0.1,  # 100 ms frames
                    possible_angles=[0, 20, 40, 60, 80, 100, 260, 280, 300, 320, 340],
                    min_event_length=2.0,
                    max_event_length=5.0
                )
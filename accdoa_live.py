import numpy as np
import pyaudio
import threading
import torch
from datetime import datetime
from queue import Queue, Empty
from collections import deque
import onnxruntime as ort
import time
import os
import gc
import warnings
import librosa

# Global audio and processing constants
FS = 24000
N_FFT = 512
HOP_LENGTH = 300
N_MICS = 4
FMIN_DOA = 50
FMAX_DOA_INIT = 4000  # initial fmax for DoA
FMAX = 9000         # cutoff frequency for spectrogram cropping
C = 343
d_max = 49 / 1000  # Maximum distance between two microphones
f_alias = 343 / (2 * d_max)  # Spatial aliasing frequency
FMAX_DOA = np.min((FMAX_DOA_INIT, FS // 2, f_alias))

N_BINS = N_FFT // 2 + 1
LOWER_BIN = int(np.floor(FMIN_DOA * N_FFT / FS))
LOWER_BIN = np.max((1, LOWER_BIN))
UPPER_BIN = int(np.floor(FMAX_DOA * N_FFT / FS))
CUTOFF_BIN = int(np.floor(FMAX * N_FFT / FS))
assert UPPER_BIN <= CUTOFF_BIN, 'Upper bin for spatial feature is higher than cutoff bin for spectrogram!'

# Normalization factor for salsa_lite --> 2*pi*f/c
DELTA = 2 * np.pi * FS / (N_FFT * C)
# Pre-compute frequency vector for use in spatial feature computation.
freq_vector = np.arange(N_BINS).astype(np.float32)
freq_vector[0] = 1  # avoid division by zero for DC
freq_vector = freq_vector[:, None, None]  # shape: (n_bins, 1, 1)

# Precompute the Hann window to speed up repeated STFT calls.
hann_window = librosa.filters.get_window('hann', N_FFT, fftbins=True)


def extract_salsalite(audio_data, normalize=True):
    """
    Extracts SALSA-lite features given multi-channel audio data.
    Audio data is expected in shape (n_mics, n_samples).
    """
    # Prepare arrays for storing STFT and log spectrograms.
    log_specs = []
    X = None

    for imic in range(N_MICS):
        # Compute the STFT using the precomputed Hann window.
        stft = librosa.stft(
            y=np.asfortranarray(audio_data[imic, :]),
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
            win_length=N_FFT,
            window=hann_window,
            pad_mode='reflect'
        )
        if X is None:
            n_frames = stft.shape[1]
            # Preallocate complex array for STFT of all microphones:
            X = np.empty((N_BINS, n_frames, N_MICS), dtype=np.complex64)
            # Preallocate log spectrogram array:
            log_specs = np.empty((N_MICS, n_frames, N_BINS), dtype=np.float32)
        X[:, :, imic] = stft

        # Compute the log power spectrum for this channel
        spec = (np.abs(stft) ** 2).T  # shape: (n_frames, n_bins)
        # librosa.power_to_db returns float32 by default; ensure correct shape.
        log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
        log_specs[imic, :, :] = log_spec

    # Normalize log power spectra over mics and frames if needed.
    if normalize:
        mean = np.mean(log_specs, axis=(0, 1), keepdims=True)
        std = np.std(log_specs, axis=(0, 1), keepdims=True)
        std[std == 0] = 1  # avoid division by zero
        log_specs = (log_specs - mean) / std

    # Compute spatial feature
    # X has shape (n_bins, n_frames, n_mics); use mic0 as reference.
    phase_vector = np.angle(X[:, :, 1:] * np.conj(X[:, :, 0, None]))
    # Normalize phase difference using delta and frequency vector.
    phase_vector = phase_vector / (DELTA * freq_vector)
    # Rearrange dimensions: (n_mics-1, n_frames, n_bins)
    phase_vector = np.transpose(phase_vector, (2, 1, 0))

    # Crop frequency bins.
    log_specs = log_specs[:, :, LOWER_BIN:CUTOFF_BIN]
    phase_vector = phase_vector[:, :, LOWER_BIN:CUTOFF_BIN]
    # Zero-out phase bins above UPPER_BIN.
    phase_vector[:, :, UPPER_BIN:] = 0

    # Stack the features along the channel axis.
    # Final shape: (n_features, n_frames, n_frequency_bins)
    audio_feature = np.concatenate((log_specs, phase_vector), axis=0)
    return audio_feature


def to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array."""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def convert_output(predictions, n_classes=3, sed_threshold=0.5):
    """Convert model output into SED mask and azimuth angles."""
    # Flatten from (T, F, 2*n_classes) to (T*F, 2*n_classes)
    predictions = predictions.reshape(-1, predictions.shape[-1])
    pred_x, pred_y = predictions[:, :n_classes], predictions[:, n_classes:]

    # SED mask: active if the magnitude exceeds the threshold.
    sed = np.sqrt(pred_x ** 2 + pred_y ** 2) > sed_threshold

    # Convert (x, y) to azimuth in degrees.
    azi = np.arctan2(pred_y, pred_x) * (180.0 / np.pi)
    azi = azi * sed  # zero-out inactive predictions

    converted_output = np.concatenate((sed, azi), axis=-1)
    return converted_output


# Suppress numpy overflow warnings (e.g., during sigmoid operations)
# warnings.filterwarnings('ignore')

# Ensure the script working directory is the same as the script's location.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print("Changing directory to:", dname)
gc.enable()
gc.collect()
n_classes = 3

# Tracking processing times
tracking_feature_ex = []
tracking_model_inf = []
tracking_processing = []

# Setup onnxruntime session.
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
ort_sess = ort.InferenceSession('./onnx_models/040225_1607_dsc_2fps_20sec_model.onnx', sess_options=sess_options)
input_names = ort_sess.get_inputs()[0].name

# Audio recording parameters.
FORMAT = pyaudio.paFloat32
CHANNELS = 4
RATE = FS
RECORD_SECONDS = 1
fpb = int(RATE * RECORD_SECONDS)  # Frames per buffer

# Use Queue (from the standard library) for thread-safe communication.
MAX_RECORDINGS = 48
data_queue = Queue(maxsize=MAX_RECORDINGS)

# Create a rolling buffer (deque) to hold the last 10 seconds (10 buffers)
rolling_audio = deque(maxlen=10)

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=fpb,
                    input_device_index=1)


def record_audio(stream, stop_event, data_queue):
    """
    Continuously record audio and put it on the queue.
    """
    while not stop_event.is_set():
        try:
            time_now = datetime.now()
            buffer = np.frombuffer(stream.read(fpb, exception_on_overflow=False), dtype=np.float32)
            data_queue.put((time_now, buffer))
        except Exception as e:
            print("Error in recording audio:", e)
            break


def infer_audio(ort_sess, data_queue):
    """
    Process one audio chunk from the queue: feature extraction, model inference, and post-processing.
    """
    try:
        # Wait for new data (blocks if the queue is empty).
        record_time, audio_buffer = data_queue.get(timeout=1)
    except Empty:
        return

    # Update rolling buffer with the new 1-second buffer.
    rolling_audio.append(audio_buffer)

    # Concatenate the rolling buffers into one long array.
    # (If fewer than 10 buffers are available, it uses what is present.)
    rolling_combined = np.concatenate(list(rolling_audio))  # shape: (num_buffers*fpb,)

    # Compute the normalization factor from the rolling window.
    # Here we compute the peak amplitude over the past 10 seconds.
    norm_factor = np.max(np.abs(rolling_combined))
    if norm_factor == 0:
        norm_factor = 1.0

    # Normalize the current 1-second audio buffer using the factor computed over the 10-second window.
    normalized_buffer = rolling_combined / norm_factor

    # Reshape the normalized buffer into (channels, samples)
    audio_data = normalized_buffer.reshape(-1, CHANNELS).T

    # Feature extraction
    feat_start = time.time()
    features = extract_salsalite(audio_data, normalize=True)
    features = features[:, -81:-1, :] # Feature shape of (7, 80, 191)
    tracking_feature_ex.append(time.time() - feat_start)

    # Model inference
    pred_start = time.time()
    input_tensor = torch.from_numpy(features).float().unsqueeze(0)
    inputs = {input_names: to_numpy(input_tensor)}
    prediction = ort_sess.run(None, inputs)
    tracking_model_inf.append(time.time() - pred_start)

    # Post-processing
    process_start = time.time()
    prediction = convert_output(prediction[0])
    avg_prediction = np.mean(prediction, axis=0)
    sed = avg_prediction[:3].astype(int)
    doa = avg_prediction[3:].astype(int)
    doa = sed * doa
    outprint = np.concatenate((sed, doa))
    print(f"[{record_time}] - {outprint}")
    tracking_processing.append(time.time() - process_start)


def main():
    """
    Main function to start the audio recording and inference threads.
    """
    stop_event = threading.Event()
    record_thread = threading.Thread(
        target=record_audio, args=(stream, stop_event, data_queue))
    record_thread.start()
    print("Threads started!")

    try:
        # Continuously infer as new audio data arrives.
        while True:
            infer_audio(ort_sess, data_queue)
    except KeyboardInterrupt:
        stop_event.set()
        record_thread.join()
        print("Recording stopped by user")
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()
    # Print tracking stats.
    print("Feature Extraction ~ N({:.4f}, {:.4f})".format(
        np.mean(tracking_feature_ex), np.var(tracking_feature_ex)))
    print("Model Inference ~ N({:.4f}, {:.4f})".format(
        np.mean(tracking_model_inf), np.var(tracking_model_inf)))
    print("Processing ~ N({:.4f}, {:.4f})".format(
        np.mean(tracking_processing), np.var(tracking_processing)))
import numpy as np
import pyaudio
import threading
import torch
from datetime import datetime
from collections import deque
import onnxruntime as ort
import time
import os
import gc
import warnings
import librosa


def extract_salsalite(audio_data, normalize=True):

    fs = 24000
    n_fft = 512
    hop_length = 300

    # Doa info
    n_mics = 4
    fmin_doa = 50
    fmax_doa = 4000

    """
    For the demo, fmax_doa = 4kHz, fs = 48kHz, n_fft = 512, hop = 300
    This results in the following:
        n_bins      = 257
        lower_bin   = 1
        upper_bin   = 42
        cutoff_bin  = 96 
        logspecs -> 95 bins total
        phasespecs -> 41 bins total

    Since these are all fixed, can we just put them into the config.yml instead
    and just read them from there and avoid these calculations
    """

    d_max = 49 / 1000 # Maximum distance between two microphones
    f_alias = 343/(2*d_max) # Spatial aliasing frequency
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

    # Convert List to NumPy Array
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


# Misc utility functions
def to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array."""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def convert_output(predictions, n_classes = 3, sed_threshold=0.5):

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
    # azi[azi < 0] += 360.0

    converted_output = np.concatenate((sed, azi), axis=-1)
    return converted_output


# For testing
extraction_time = []

#suppress warnings for numpy overflow encountered in exp function during sigmoid
warnings.filterwarnings('ignore')

# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print("Changing directory to : ", dname)
gc.enable()
gc.collect()
n_classes = 3

# Tracking the virtual memory consumption of the device
tracking_feature_ex = []
tracking_model_inf = []
tracking_processing = []

# Setup onnxruntime
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
ort_sess = ort.InferenceSession('./onnx_models/030225_0913_btndsc_2fps_model.onnx', sess_options=sess_options)
input_names = ort_sess.get_inputs()[0].name

# Global variables
CHUNK = 500
FORMAT = pyaudio.paFloat32
CHANNELS = 4
RATE = 24000
MAX_RECORDINGS = 48
INPUT_DEVICE_INDEX = 1
RECORD_SECONDS = 1
fpb = int(RATE * RECORD_SECONDS) # Frames per buffer

# Queue to store audio buffers 
data_queue = deque()

# Stream
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=fpb,
                    input_device_index=INPUT_DEVICE_INDEX)

lock = threading.Lock()

# Function to record audio
def record_audio(stream, stop_event):

    global data_queue # Bufer queue for all the data

    while not stop_event.is_set():
        try:
            # Read the data from the buffer as np.float32
            time_now = datetime.now()
            buffer = np.frombuffer(stream.read(fpb, exception_on_overflow=False), dtype=np.float32)

            # Append the audio data and recording time into the buffer queues
            data_queue.append((time_now, buffer))
        except Exception as e:
            print("Something went wrong!")
            print(e)
            break


def infer_audio(ort_sess):

    global data_queue

    # Wait until there is something in the buffer queue
    while len(data_queue) == 0:
        time.sleep(0.01)  # Prevent busy waiting

    # We take the latest (most recent) item and copy it in order to make modifications on the data
    all_data = data_queue.popleft()
    record_time = all_data[0] # No need copy for string data, apparently
    audio_data = all_data[1].copy() # Float data needs to be copied
    peak = np.max(np.abs(audio_data))
    audio_data = audio_data / peak
    audio_data = audio_data.reshape(-1,4).T

    # Feature extraction
    feat_start = time.time()
    features = extract_salsalite(audio_data, normalize=True) # Shape of (7, 81, 191)
    features = features[:, :-1, :]
    tracking_feature_ex.append(time.time() - feat_start)

    # Model prediction
    pred_start = time.time()
    input_tensor = torch.from_numpy(features).type(torch.FloatTensor).unsqueeze(0)
    inputs = {input_names: to_numpy(input_tensor)}
    prediction = ort_sess.run(None, inputs)
    tracking_model_inf.append(time.time() - pred_start)

    # Basic prediction post-processing functions
    process_start = time.time()
    prediction = convert_output(prediction[0])
    avg_predicion = np.mean(prediction, axis=0)
    sed = avg_predicion[:3].astype(int)
    doa = avg_predicion[3:].astype(int)
    doa = sed * doa
    outprint = np.concatenate((sed, doa))
    print("[{}] - {}".format(record_time, outprint))
    tracking_processing.append(time.time()-process_start)



def main():
    """Main function to do concurrent recording and inference""" 
    global ort_sess

    # Create an event to signal the threads to stop
    stop_event = threading.Event()
    record_thread = threading.Thread(target=record_audio,
                                    args=(stream,stop_event))
    record_thread.start()
    print("Threads started!")

    try:
        while True:
            infer_audio(ort_sess)
    except KeyboardInterrupt:
        # Signal the threads to stop
        stop_event.set()

        # Wait for the threads to finish
        record_thread.join()
        print("Recording stopped by user")

        # End the stream gracefully
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":

    # Main recording and inference function
    main()

    print("Feature Extraction ~ N({:.4f}, {:.4f})".format(np.mean(np.array(tracking_feature_ex)),
                                                          np.var(np.array(tracking_feature_ex))))

    print("Model Inference ~ N({:.4f}, {:.4f})".format(np.mean(np.array(tracking_model_inf)),
                                                          np.var(np.array(tracking_model_inf))))

    print("Processing ~ N({:.4f}, {:.4f})".format(np.mean(np.array(tracking_processing)),
                                                          np.var(np.array(tracking_processing))))
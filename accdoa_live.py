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
from feature_label_extraction import extract_salsalite

# Misc utility functions
def to_numpy(tensor):
    """Convert the feature tensor into np.ndarray format for the ONNX model to run 

    Inputs
        tensor (PyTorch Tensor) : input PyTorch feature tensor of any shape 

    Returns
        tensor (PyTorch Tensor) : The same tensor, but in np.ndarray format to input into ONNX model
    """
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
    azi[azi < 0] += 360.0
    
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
os.system('cls')
print("Changing directory to : ", dname)
print("Screen cleared!")
gc.enable()
gc.collect()
n_classes = 3

# Tracking the virtual memory consumption of the device
tracking_memory = []
tracking_feature_ex = []
tracking_model_inf = []
tracking_processing = []

# Moving average filter of 5 windows with 4 classes
moving_sed = []
moving_doa = []

for i in range(5):
    starting_zeros = [0] * n_classes
    moving_sed.append(starting_zeros)
    moving_doa.append(starting_zeros)


# Setup onnxruntime
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
ort_sess = ort.InferenceSession('./onnx_models/240125_1519_full_highsnr_model.onnx', sess_options=sess_options)
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
    """Define the data buffer queue outside of the function and call it globally.
    This function is used to record audio data and push them to the 
    buffer queue for another function to use for inference. 
    
    Inputs:
        stream (pyaudio.stream) : Stream class defined by PyAudio
        stop_event (thread) : Thread to indicate whether this thread should continue running
        
    Returns:
        None
    """

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
    """Define the data buffer queue outside of the function. In fact, may not even need to 
    append the recording time into the buffer if do not wish to keep track of time. It was
    used to keep track of when the audio data was recorded and to make sure that the system
    is inferring on the correct audio data in the queue.

    This function is used to use an ONNXRUNTIME session in order to infer on the audio data
    recorded. Potentially, this function can be modified to return the inference output itself
    to pass to another function/system for post-processing.

    Inputs:
        ort_sess (onnxruntime session) : The onnxruntime session of our model for inference

    Returns:
        None
    """

    global data_queue, moving_sed, moving_doa, recent_rms, b, a, filter_states

    # Wait until there is something in the buffer queue
    while len(data_queue) == 0:
        time.sleep(0.01)  # Prevent busy waiting

    # We take the latest (most recent) item and copy it in order to make modifications on the data
    all_data = data_queue.popleft()
    record_time = all_data[0] # No need copy for string data, apparently
    audio_data = all_data[1].copy() # Float data needs to be copied
    audio_data = audio_data.reshape(-1,4).T

    # Feature extraction
    feat_start = time.time()
    features = extract_salsalite(audio_data) # Shape of (7, 81, 191)
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
    print("[{}] - {}".format(record_time, prediction))
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
import pyaudio, wave, os, time

def recorder(recording_filename = '', recording_device_index = None, recording_length = 10, recording_sample_frequency = None, buffer_length = 1,
             playback_filename = '', playback_device_index = None, verbose = False):
    """
    Function that allows one to record and/or play back files simultaneously using pyaudio. There are three modes: Playback-only, recording-only, and playback-while-recording.
    The desired mode is determined by whether the playback file name and recording file names are given as empty strings or not.
    
    Playback-only mode is activated if playback_filename is non-empty and recording_filename is empty.
    It will play the audio file specified in playback_filename using tbe device in playback_device_index,
    with the native number of channels and sampling frequency specified in playback_filename.
    
    Recording-only mode is activated if recording_filename is non-empty and playback_filename is empty.
    It will record an audio file of recording_length seconds at recording_sample_frequency to recording_filename using the device in recording_device_index.
    
    Playback-while-recording mode is activated if both playback_filename and recording_filename are non-empty.
    It will record an audio file of recording_length + 2*buffer_length seconds at recording_sample_frequency to recording_filename using the device in recording_device_index.
    Recording will start for buffer_length seconds before playback starts, and will end buffer_length seconds after playback ends.

    ========
     Inputs
    ========

    recording_filename (str): Specifies the name (or filepath) of the .wav file that the recording will be written in.
                              Will create the file and associated directories if it doesn't already exist, and overwrites the file if it already exists.
                              If it is an empty string, then no recording will occur (use this to toggle between recording-only andplayback-only mode).
    recording_device_index (int): Specifies the index of the device that you want to use for recording.
                                  If None, then the default recording (input) device of the system will be used.
                                  See the Help section for more details.
    recording_length (float): The length of the recording in seconds. The default value is 10, giving a recording length of 10 seconds in recording-only mode,
                              and the length of the playback file in playback_filename PLUS 2*buffer_length seconds in recording-while-playback mode.
    recording_sample_frequency (int): The sampling frequency desired for the recording, in Hz. Default value is the default sampling frequency of the device in recording_device_index.
                                      In other words, if recording_device_index == None, then the default value is the default sampling frequency of the default input device.
                                      If manually setting, make sure that this value is compatible with your sound card and microphone/input device, otherwise PyAudio will throw an error.
    buffer_length (float): The length of the buffer in seconds between the start of recording and playback, as well as the end of playback and recording.
                           Is only used in playback-while-recording mode.
    playback_filename (str): Specifies the name (or filepath) of the file that you want to play over the speakers.
                             If it is an empty string, then no playback will occur (you can use this to toggle between recording-only and playback-only mode).
    playback_device_index (int): Specifies the index of the device that you want to use for playback. If None, then the default playback (output) device of the system will be used.
                                 See the Help section for more details.
    verbose (bool): If True, prints out what the turntable is doing and if commands are executed properly. If False, prints out nothing.

    Note that if both playback_filename and recording_filename are empty strings, then this function does nothing.

    =========
     Outputs
    =========

    None

    ======
     Help
    ======

    To obtain playback_device_index and recording_device_index for the desired playback/recording devices, use the following code:

    import pyaudio
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print (i, p.get_device_info_by_index(i).get('name'))

    Also, note that the playback-and-recording mode of this function can also be done in Audacity, if the 'overdub' function is enabled (Edit -> Preferences -> Recording).
    However, in practice, this code is combined with the turntable, so we need to do everything in Python.
    
    Debugging note: The most likely culprit for recording failures is possibly the value of chunk_size, which can be increased manually if necessary.
    """

    p = pyaudio.PyAudio() # Define a new pyaudio object for handling the recording and playback of files.
#%%
    if len(playback_filename) > 0 and len(recording_filename) == 0: # This specifies playback-only mode.
        data_play = wave.open(playback_filename, 'rb') # Open the wave file specified in playback_filename and retrieve its associated metadata. The 'rb' argument refers to read-only mode.

        if verbose:
            print('Playback-only mode selected.')
            print(playback_filename + ' was loaded with {:d} channels at a sampling frequency of {:.0f} Hz.'.format(data_play.getnchannels(),data_play.getframerate()))

            channel_warning_string = 'Warning : Number of output channels ({:d}) of device exceeds number of channels ({:d}) in playback file. This may lead to distortions in playback or errors due to pyaudio.'
            if playback_device_index == None:
                if p.get_default_output_device_info()['maxOutputChannels'] < data_play.getnchannels():
                    print(channel_warning_string.format(p.get_default_output_device_info()['maxOutputChannels'],data_play.getnchannels()))
            else:
                if p.get_device_info_by_index(playback_device_index)['maxOutputChannels'] < data_play.getnchannels():
                    print(channel_warning_string.format(p.get_device_info_by_index(playback_device_index)['maxOutputChannels'],data_play.getnchannels()))


        def callback(in_data, frame_count, time_info, flag): # Define pyaudio callback object (this will be activated to load new frames into the buffer when it is empty).
            data = data_play.readframes(frame_count)
            return (data, pyaudio.paContinue)

        # Define a new output stream called playback_stream (similar to MATLAB audiorecorder object).
        playback_stream = p.open(format = p.get_format_from_width(data_play.getsampwidth()),
                                 channels = data_play.getnchannels(), # This is the number of channels of audio in playback_filename. Must be below the maximum number of output channels allowed for the particular device
                                 rate = data_play.getframerate(), # This is the sampling frequency of audio in playback_filename.
                                 output = True, # Specifies that this stream is an output stream, i.e. the one that we want to use for playback.
                                 stream_callback = callback, # Specifies the callback function to load more frames into the stream when buffer is emptied.
                                 output_device_index = playback_device_index) # Remember to match the output device according to the list in p.get_device_info_by_index if manually changing.

        # Start the playback (i.e. output) stream.
        playback_stream.start_stream() # Starts the playback stream. This line is the one that plays back the audio.
        if verbose:
            print('Playback started.')

        while playback_stream.is_active(): # This loop waits until the stream ends (i.e. until the file finishes playing) before stopping it.
                                           # Callback function is continuously called while the stream is active.
            time.sleep(0.1)

        playback_stream.stop_stream() # Stops the playback (i.e. output) stream.
        if verbose:
            print('Playback ended.')
        playback_stream.close() # Streams must be closed when they are done being used, to prevent errors in future playback.
        data_play.close()
        if verbose:
            print('Playback (output) stream closed.')
        p.terminate() # Close pyaudio.
        if verbose:
            print('PyAudio closed.')
#%%
    elif len(playback_filename) == 0 and len(recording_filename) > 0: # This specifies recording-only mode.
        if verbose:
            print('Recording-only mode selected.')

        if recording_device_index == None: # then we will use the default input device.
            recording_device_index = p.get_default_input_device_info()['index']

        if recording_sample_frequency == None: # then we will use the default sample rate of the device in recording_device_index.
            recording_sample_frequency = int(p.get_device_info_by_index(recording_device_index)['defaultSampleRate']) # The value obtained from the function call is a float, so we need to typecast it to int first.

        frames = [] # Initialise the list of values that we want to store the recorded data in.
        chunk_size = 1 # chunk_size is the number of data points to store in the buffer at a time before writing to frames.
                       # We set it to 1 for lowest latency.
                       # The default value for PyAudio.open() is 1024, and should not cause issues in most systems,
                       # so we have not specified it as a parameter in the recorder function. However, feel free to
                       # change this value within the function for future debugging purposes if there are buffer overflow
                       # problems or samples dropping etc.
        nchunks = int(recording_sample_frequency / chunk_size * recording_length)     # Since we read data in blocks of chunk_size, the number of chunks (nchunks)
                                                                                      # that we need to read is calculated by taking the total number of samples
                                                                                      # divided by chunk_size. The total number of samples to take is, of course,
                                                                                      # recording_sample_frequency * recording_length.
                                                                                      # We could have added 1 to the number of chunks to ensure that the recording is at least
                                                                                      # recording_length seconds long, although that is not done here.
        nchannels = p.get_device_info_by_index(recording_device_index)['maxInputChannels'] # This is the number of channels used for recording. We just use
                                                                                           # the maximum number of input (recording) channels that are present
                                                                                           # in the recording device.
                                                                                           # Can be changed in the future to any number less than that to save memory space too.

        # Define a new input stream called recording_stream (similar to MATLAB audiorecorder object).
        recording_stream = p.open(format = pyaudio.paInt16, # Specifies the recording format to be 16-bit integers
                                  channels = nchannels, 
                                  rate = recording_sample_frequency, # Record at the rate specified by recording_sample_frequency.
                                  input = True, # Specifies that this stream is an input stream, i.e. the one that we want to use for recording.
                                  frames_per_buffer = chunk_size, # Tell PyAudio that we want to record in chunks of chunk_size.
                                  input_device_index = recording_device_index) # Remember to match the input device according to the list in p.get_device_info_by_index if manually changing.

        if verbose:
            print('Recording stream opened with {} channels at a sampling frequency of {} Hz.'.format(nchannels, recording_sample_frequency))
            print('Recording will last for {} seconds.'.format(recording_length))

        try: # We use a try-except statement for the recording part because we don't want to save any recorded data even if there are any errors with the code/recording.
            time_elapsed = 0 # Initialise the time elapsed to 0 (seconds). This value will only be used in verbose mode to print the current recording status.

            # Start the recording.
            for i in range(nchunks): # Read data for nchunks number of times with each block having chunk_size amount of data.
                data = recording_stream.read(chunk_size) # Read (i.e. record) a chunk_size block of data from recording_stream.
                frames.append(data) # Add the read (i.e. recorded) data to the list (frames) that we initialised previously.
                if verbose: # Then print the time elapsed each second.
                    if i * chunk_size  / recording_sample_frequency > time_elapsed + 1: # i*chunk_size/recording_sample_frequency is the total amount of time elapsed so far.
                                                                                        # So if that is greater than time_elapsed + 1, it means that 1 second has elapsed.
                            time_elapsed += 1
                            print('Time elapsed : {} seconds.'.format(time_elapsed) + chr(127) * 10, end = '\r') # We print the time elapsed (rewriting the line each second due
                                                                                                                # to the carriage return \r) every second. The delete characters
                                                                                                                # (denoted by chr(127)) are there to erase any possible
                                                                                                                # extra characters to ensure a clean output.
                                                                                                                # See https://realpython.com/python-print/ for more details on
                                                                                                                # animations with the Python print function.

            recording_stream.stop_stream() # Stops the recording (i.e. input) stream.
            if verbose:
                print('Recording ended.' + chr(127) * 10) # We add the delete characters (chr(127)) again to ensure a clean output.

            recording_stream.close() # Streams must be closed when they are done being used, to prevent errors in future recording.
            if verbose:
                print('Recording (input) stream closed.')
        except KeyboardInterrupt:
            print('Function has been terminated by user. Stopping recording and outputting data acquired until now, if any.')
        except MemoryError:
            print('Memory is full. Stopping recording and outputting data acquired until now, if any.')   
        except: # If any other error occurs...
            raise # raise it (i.e. print the error text)
        finally:
            if verbose:
                print('Writing recorded data to ' + recording_filename)
                
            recording_folder = ''.join([i + os.sep for i in recording_filename.split(os.sep)[:-1]]) # Remove the ~.wav filename part (last element of recording_filename.split) from the path provided to create the directory.
            
            if len(recording_folder) > 0 and not os.path.exists(recording_folder): # If the folder specified in the file path doesn't exist, then create it.
                os.makedirs(recording_folder)
                
            data_rec = wave.open(recording_filename, 'wb') # Open a wave stream to start writing data into the wave file.
            data_rec.setnchannels(nchannels)
            data_rec.setsampwidth(p.get_sample_size(pyaudio.paInt16)) # We want to output in 16-bit float format, so we set the sample width to 2 bytes = 16 bits.
            data_rec.setframerate(recording_sample_frequency)
            data_rec.writeframes(b''.join(frames)) # The letter b before the string indicates byte-type values in frames. Since we read in blocks of chunk_size, the final
                                                   # .wav file that is written will be slightly longer than the desired timing in seconds.
            data_rec.close()
            if verbose:
                print('Finished writing recorded data to ' + recording_filename)

            p.terminate() # Close pyaudio at the very end. Notice the first-in-last-out policy of closing streams here.
            if verbose:
                print('PyAudio closed.')
#%%
    elif len(playback_filename) > 0 and len(recording_filename) > 0: # This specifies playback-while-recording mode.
        if verbose:
            print('Playback-while-recording mode selected.')
            
        # Initialisation of playback parameters 
        data_play = wave.open(playback_filename, 'rb') # Open the wave file specified in playback_filename and retrieve its associated metadata. The 'rb' argument refers to read-only mode.
        if verbose:
            print(playback_filename + ' was loaded with {:d} channels at a sampling frequency of {:.0f} Hz.'.format(data_play.getnchannels(),data_play.getframerate()))

            channel_warning_string = 'Warning : Number of output channels ({:d}) of device exceeds number of channels ({:d}) in playback file. This may lead to distortions in playback or errors due to pyaudio.'
            if playback_device_index == None:
                if p.get_default_output_device_info()['maxOutputChannels'] < data_play.getnchannels():
                    print(channel_warning_string.format(p.get_default_output_device_info()['maxOutputChannels'],data_play.getnchannels()))
            else:
                if p.get_device_info_by_index(playback_device_index)['maxOutputChannels'] < data_play.getnchannels():
                    print(channel_warning_string.format(p.get_device_info_by_index(playback_device_index)['maxOutputChannels'],data_play.getnchannels()))

        def callback(in_data, frame_count, time_info, flag): # Define pyaudio callback object (this will be activated to load new frames into the buffer when it is empty).
            data = data_play.readframes(frame_count)
            return (data, pyaudio.paContinue)

        # Initialisation of recording parameters.
        if recording_device_index == None: # then we will use the default input device.
            recording_device_index = p.get_default_input_device_info()['index']

        if recording_sample_frequency == None: # then we will use the default sample rate of the device in recording_device_index.
            recording_sample_frequency = int(p.get_device_info_by_index(recording_device_index)['defaultSampleRate']) # The value obtained from the function call is a float, so we need to typecast it to int first.

        recording_length = data_play.getnframes()/data_play.getframerate() # Set the length of the recording to the length of the playback file.

        frames = [] # Initialise the list of values that we want to store the recorded data in.
        chunk_size = 1 # chunk_size is the number of data points to store in the buffer at a time before writing to frames.
                       # We set it to 1 for lowest latency.
                       # The default value for PyAudio.open() is 1024, and should not cause issues in most systems,
                       # so we have not specified it as a parameter in the recorder function. However, feel free to
                       # change this value within the function for future debugging purposes if there are buffer overflow
                       # problems or samples dropping etc.
        buffer_chunks = int(recording_sample_frequency / chunk_size * buffer_length) # This is the number of blocks to read for the buffer (calculation has same idea as nchunks).
        nchunks = int(recording_sample_frequency / chunk_size * recording_length)     # Since we read data in blocks of chunk_size, the number of chunks (nchunks)
                                                                                      # that we need to read is calculated by taking the total number of samples
                                                                                      # divided by chunk_size. The total number of samples to take is, of course,
                                                                                      # recording_sample_frequency * recording_length.
                                                                                      # We could have added 1 to the number of chunks to ensure that the recording is at least
                                                                                      # recording_length seconds long, although that is not done here.
        nchannels = p.get_device_info_by_index(recording_device_index)['maxInputChannels'] # This is the number of channels used for recording. We just use
                                                                                           # the maximum number of input (recording) channels that are present
                                                                                           # in the recording device.
                                                                                           # Can be changed in the future to any number less than that to save memory space too.

        # Define a new input stream called recording_stream (similar to MATLAB audiorecorder object).
        recording_stream = p.open(format = pyaudio.paInt16, # Specifies the recording format to be 16-bit integers
                                  channels = nchannels, 
                                  rate = recording_sample_frequency, # Record at the rate specified by recording_sample_frequency.
                                  input = True, # Specifies that this stream is an input stream, i.e. the one that we want to use for recording.
                                  frames_per_buffer = chunk_size, # Tell PyAudio that we want to record in chunks of chunk_size.
                                  input_device_index = recording_device_index) # Remember to match the input device according to the list in p.get_device_info_by_index if manually changing.

        if verbose:
            print('Recording stream opened with {} channels at a sampling frequency of {} Hz.'.format(nchannels, recording_sample_frequency))
            print('Buffer of {} seconds will be added to start and end of playback.'.format(buffer_length))

        try: # We use a try-except statement for the recording part because we don't want to save any recorded data even if there are any errors with the code/recording.
            time_elapsed = 0 # Initialise the time elapsed to 0 (seconds). This value will only be used in verbose mode to print the current recording status.

            # Start the recording.
            for i in range(nchunks + 2*buffer_chunks): # First read data for buffer_chunks number of times with each block having chunk_size amount of data.
                data = recording_stream.read(chunk_size) # Read (i.e. record) a chunk_size block of data from recording_stream.
                frames.append(data) # Add the read (i.e. recorded) data to the list (frames) that we initialised previously.
                if verbose: # Then print the time elapsed each second.
                    if i * chunk_size  / recording_sample_frequency > time_elapsed + 1: # i*chunk_size/recording_sample_frequency is the total amount of time elapsed so far.
                                                                                        # So if that is greater than time_elapsed + 1, it means that 1 second has elapsed.
                            time_elapsed += 1
                            print('Time elapsed : {} seconds.'.format(time_elapsed) + chr(127) * 10, end = '\r') # We print the time elapsed (rewriting the line each second due
                                                                                                                # to the carriage return \r) every second. The delete characters
                                                                                                                # (denoted by chr(127)) are there to erase any possible
                                                                                                                # extra characters to ensure a clean output.
                                                                                                                # See https://realpython.com/python-print/ for more details on
                                                                                                                # animations with the Python print function.
                    if i == nchunks + buffer_chunks:
                        print('Playback ended.' + chr(127) * 15)
                if i == buffer_chunks: # Once the first buffer_chunks blocks have been recorded, we start the playback.
                    # Define a new output stream called playback_stream (similar to MATLAB audiorecorder object).
                    playback_stream = p.open(format = p.get_format_from_width(data_play.getsampwidth()),
                                 channels = data_play.getnchannels(), # This is the number of channels of audio in playback_filename. Must be below the maximum number of output channels allowed for the particular device
                                 rate = data_play.getframerate(), # This is the sampling frequency of audio in playback_filename.
                                 output = True, # Specifies that this stream is an output stream, i.e. the one that we want to use for playback.
                                 stream_callback = callback, # Specifies the callback function to load more frames into the stream when buffer is emptied.
                                 output_device_index = playback_device_index) # Remember to match the output device according to the list in p.get_device_info_by_index if manually changing.
                    if verbose:
                        print('Playback started.' + chr(127) * 10)

            # Start the playback (i.e. output) stream.
            playback_stream.start_stream() # Starts the playback stream. This line is the one that plays back the audio.
            
            playback_stream.close() # Streams must be closed when they are done being used, to prevent errors in future playback.
            data_play.close()
            if verbose:
                print('Playback (output) stream closed.')

            recording_stream.stop_stream() # Stops the recording (i.e. input) stream.
            if verbose:
                print('Recording ended.' + chr(127) * 10) # We add the delete characters (chr(127)) again to ensure a clean output.

            recording_stream.close() # Streams must be closed when they are done being used, to prevent errors in future recording.
            if verbose:
                print('Recording (input) stream closed.')
        except KeyboardInterrupt:
            print('Function has been terminated by user. Stopping recording and outputting data acquired until now, if any.')
        except MemoryError:
            print('Memory is full. Stopping recording and outputting data acquired until now, if any.')   
        except: # If any other error occurs...
            raise # raise it (i.e. print the error text)
        finally:
            if verbose:
                print('Writing recorded data to ' + recording_filename)

            recording_folder = ''.join([i + os.sep for i in recording_filename.split(os.sep)[:-1]]) # Remove the ~.wav filename part (last element of recording_filename.split) from the path provided to create the directory.

            if len(recording_folder) > 0 and not os.path.exists(recording_folder): # If the folder specified in the file path doesn't exist, then create it.
                os.makedirs(recording_folder)

            data_rec = wave.open(recording_filename, 'wb') # Open a wave stream to start writing data into the wave file.
            data_rec.setnchannels(nchannels)
            data_rec.setsampwidth(p.get_sample_size(pyaudio.paInt16)) # We want to output in 16-bit float format, so we set the sample width to 2 bytes = 16 bits.
            data_rec.setframerate(recording_sample_frequency)
            data_rec.writeframes(b''.join(frames)) # The letter b before the string indicates byte-type values in frames. Since we read in blocks of chunk_size, the final
                                                   # .wav file that is written will be slightly longer than the desired timing in seconds.
            data_rec.close()
            if verbose:
                print('Finished writing recorded data to ' + recording_filename)

            p.terminate() # Close pyaudio at the very end. Notice the first-in-last-out policy of closing streams here.
            if verbose:
                print('PyAudio closed.')
#%%
    else: # This means that both playback_filename and recording_filename are empty strings.
        if verbose:
            print('No playback or recording filenames specified. Program terminating.')


def main():
    # File paths for playback and recording.
    playback_file = './demo_audio/generated_track.wav'
    recording_file = './demo_audio/track_genrec.wav'
    
    # Device indices (None means the default device is used)
    recording_device_index = None
    playback_device_index = None

    # Additional parameters
    buffer_length = 1       # Buffer of 1 second before and after playback in playback-while-recording mode.
    verbose = True          # Set to True to see detailed status output

    # Inform the user
    print("Starting playback and simultaneous recording...")
    print("Playback file: {}".format(playback_file))
    print("Recording file: {}".format(recording_file))
    
    # Call the recorder function in playback-while-recording mode.
    recorder(
        recording_filename=recording_file,
        recording_device_index=recording_device_index,
        playback_filename=playback_file,
        playback_device_index=playback_device_index,
        buffer_length=buffer_length,
        verbose=verbose
    )
    
    print("Playback and recording completed.")
    print("Recorded file saved as:", recording_file)

if __name__ == "__main__":
    main()
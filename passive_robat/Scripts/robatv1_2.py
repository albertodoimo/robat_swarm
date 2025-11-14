
# Robat v0 code to use with different arrays(PDM passive array or I2S array).
#
# FEATURES:
# > Compute pra alg for angle detection
# > Calculates dB values 
# > Triggers computation of angle only over threshold
# > Save recordings every x minutes 
# > Save file data from angle/polar plot in csv and xml files

print('import libraries...')

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal 
import sounddevice as sd
import soundfile as sf
import argparse
import queue
import datetime
import time
import random
import os
import threading
import pandas as pd 
import netifaces as ni
import csv

from thymiodirect import Connection 
from thymiodirect import Thymio

from functions.get_card import get_card 
from functions.save_data_to_csv import save_data_to_csv
from functions.utilities import pascal_to_dbspl, calc_native_freqwise_rms, interpolate_freq_response
from functions.save_data_to_xml import save_data_to_xml
from functions.matched_filter import matched_filter
from functions.detect_peaks import detect_peaks

from AudioProcessor import AudioProcessor
from RobotMove import RobotMove
from shared_queues import angle_queue, level_queue
print('imports done')

#################################################################################################
# SETUP PARAMETERS

# Create queues for storing data
timestamp_queue = queue.Queue()

timestamp_bool = True  # Set to True to save timestamps, False to not save timestamps
if timestamp_bool == True:
    print('Timestamps saving is ENABLED')
else:
    print('Timestamps saving is DISABLED')

recording_bool = True  # Set to True to record audio, False to just process audio without recording
if recording_bool == True:
    print('Recording is ENABLED')
else:
    print('Recording is DISABLED')


# Get the index of the USB card
usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)

ni.ifaddresses('wlan0')
raspi_local_ip = ni.ifaddresses('wlan0')[2][0]['addr']
print('raspi_local_ip =', raspi_local_ip)

################################################################################################
# EXPERIMENT PARAMETERS

behavior = 'repulsion'  # Options: 'attraction', 'repulsion', 'dynamic_movement'

# Parameters for the DOA alga'  # Optorithms70
trigger_level =  70 # dB SPL
critical_level = 80 # dB SPL
c = 343   # speed of sound
fs = 48000

rec_samplerate = 48000
input_buffer_time = 0.04 # seconds
block_size = int(input_buffer_time*fs)  #used for the shared queue from which the doa is computed, not anymore for the output stream
channels = 5
mic_spacing = 0.018 #m
nfft = 512
ref = channels//2 #central mic in odd array as ref
#print('ref=',ref) 
#ref= 0 #left most mic as reference
critical = []

# Possible algorithms for computing DOA:CC, DAS
method = 'DAS'

# Parameters for the CC algorithm
avar_theta = None
theta_values   = []

# # Parameters for the PRA algorithm
# echo = pra.linear_2D_array(center=[(channels-1)*mic_spacing//2,0], M=channels, phi=0, d=mic_spacing)

# Parameters for the DAS algorithm
theta_das = np.linspace(-90, 90, 61) # angles resolution for DAS spectrum
N_peaks = 1 # Number of peaks to detect in DAS spectrum

# Parameters for the chirp signal
duration_out = 10e-3  # Duration in seconds

if behavior == 'attraction':
    silence_post = 10 # [ms] can probably pushed to 20
elif behavior == 'repulsion':
    silence_post = 10 # [ms] can probably pushed to 20
elif behavior == 'dynamic_movement':
    silence_post = 110 # [ms] can probably pushed to 20

if behavior == 'attraction':
    amplitude = 0 # Amplitude of the chirp
elif behavior == 'repulsion':
    amplitude = 0 # Amplitude of the chirp
elif behavior == 'dynamic_movement':
    amplitude = 0.5 # Amplitude of the chirp

t = np.linspace(0, duration_out, int(fs*duration_out))
start_f, end_f = 24e3, 2e3
sweep = signal.chirp(t, start_f, t[-1], end_f)
sweep *= signal.windows.tukey(sweep.size, 0.2)
sweep *= 0.8

silence_samples_post = int(silence_post * fs/1000)
silence_vec_post = np.zeros((silence_samples_post, ))
post_silence_sig = np.concatenate((sweep, silence_vec_post))
full_sig = post_silence_sig

stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])
data = amplitude * np.float32(stereo_sig)

out_blocksize = int(len(data))  # Length of the output signal
print('out_blocksize =', out_blocksize)

# plot and save data 
# plt.figure(figsize=(10, 4))
# plt.plot(np.arange(len(full_sig)) / fs, data[:, 0], label='Left Channel')
# plt.plot(np.arange(len(full_sig)) / fs, data[:, 1], label='Right Channel')
# plt.title('Chirp Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.tight_layout()
# plt.savefig('chirp_signal.png')
# plt.close()

# Calculate highpass and lowpass frequencies based on the array geometry
auto_hipas_freq = int(343/(2*(mic_spacing*(channels-1))))
print('HP frequency:', auto_hipas_freq)
auto_lowpas_freq = int(343/(2*mic_spacing))
print('LP frequency:', auto_lowpas_freq)
highpass_freq, lowpass_freq = [auto_hipas_freq ,auto_lowpas_freq]
#highpass_freq, lowpass_freq = [200 ,20e3]
freq_range = [start_f, end_f]

cutoff = auto_hipas_freq # [Hz] highpass filter cutoff frequency
sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')

# Set the path for the sensitivity CSV file relative to the current script location
script_dir = os.path.dirname(os.path.abspath(__file__))
sensitivity_path = os.path.join(script_dir, 'Knowles_SPH0645LM4H-B_sensitivity.csv')
sensitivity = pd.read_csv(sensitivity_path)

analyzed_buffer_time = 0.01
block_size_analyzed_buffer = int(analyzed_buffer_time * fs)

in_sig = np.zeros(block_size_analyzed_buffer, dtype=np.float32)  # Initialize the buffer for the audio input stream
print('in_sig shape:', np.shape(in_sig))
centrefreqs, freqrms = calc_native_freqwise_rms(in_sig, fs)

freqs = np.array(sensitivity.iloc[:, 0])  # first column contains frequencies
sens_freqwise_rms = np.array(sensitivity.iloc[:, 1])  # Last column contains sensitivity values 
interp_sensitivity = interpolate_freq_response([freqs, sens_freqwise_rms],
                centrefreqs)

frequency_band = [2e3, 20e3] # min, max frequency to do the compensation Hz
tgtmic_relevant_freqs = np.logical_and(centrefreqs>=frequency_band[0],
                            centrefreqs<=frequency_band[1])
# Thymio movement parameters

# Straight speed
speed = 200 
turn_speed = 200

left_sensor_threshold = 250
right_sensor_threshold = 250	

#######################################################################################################
# MAIN FUNCTION

if __name__ == '__main__':

    def int_or_str(text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    # Parse commandline arguments to configure the interface for a simulation (default = real robot)
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                    'If no arguments are given, the code will run with a real Thymio.', add_help=False)
    # Add optional arguments
    parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        'filename', nargs='?', metavar='FILENAME',
        help='audio file to store recording to')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    parser.add_argument(
        '-c', '--channels', type=int, default=channels, help='number of input channels')
    parser.add_argument(
        '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
    args = parser.parse_args(remaining)

    # Parse arguments and pass them to main function
    args = parser.parse_args()
    args.buffer = np.zeros((block_size, channels))

    # Set initial parameters for the audio processing
    startime = datetime.datetime.now()
    args.samplerate = fs
    args.rec_samplerate = rec_samplerate
    args.angle = 0

    # Set general save path
    time1 = startime.strftime('%Y-%m-%d')
    save_path = '/home/thymio/robat_py/robat_swarm/passive_robat/'
    folder_name = str(raspi_local_ip)
    folder_path = os.path.join(save_path,'Data', folder_name, time1)

    if recording_bool == True:
        # Create folder for saving recordings
        time2 = startime.strftime('_%Y-%m-%d__%H-%M-%S')
        os.makedirs(folder_path, exist_ok=True)
        
        name = 'MULTIWAV_' + str(raspi_local_ip) + str(time2) + '.wav'
        args.filename = os.path.join(folder_path, name)

        if args.samplerate is None:  
            print('error!: no samplerate set! Using default')
            device_info = sd.query_devices(args.device, 'input')
            args.samplerate = int(device_info['default_samplerate'])
        if args.filename is None:
            timenow = datetime.datetime.now()
            args.filename = name
        print(args.samplerate)
    
    if timestamp_bool == True:
        # Create folder for saving timestamps
        time2 = startime.strftime('_%Y-%m-%d__%H-%M-%S')
        os.makedirs(folder_path, exist_ok=True)
        folder_path_data = os.path.join(save_path, 'Data', folder_name, time1)
        os.makedirs(folder_path_data, exist_ok=True)

        name = 'TIMESTAMPS_' + str(raspi_local_ip) + str(time2) + '.csv'
        args.timestamp_filename = os.path.join(folder_path_data, name)

    # Create instances of the AudioProcessor and RobotMove classes
    audio_processor = AudioProcessor(fs, channels, block_size, analyzed_buffer_time, data, args, trigger_level, critical_level, mic_spacing, ref, highpass_freq, lowpass_freq, theta_das, N_peaks,
                                      usb_fireface_index, args.subtype, interp_sensitivity, tgtmic_relevant_freqs, args.filename, args.rec_samplerate, sos, sweep)
    robot_move = RobotMove(speed, turn_speed, left_sensor_threshold, right_sensor_threshold, critical_level, trigger_level, ground_sensors_bool = True)
    
    # Create threads for the audio input and recording
    if recording_bool == True:
        inputstream_thread = threading.Thread(target=
            audio_processor.continuos_recording, daemon = True)
        inputstream_thread.start()
    else:
        inputstream_thread = threading.Thread(target=
            audio_processor.input_stream,  daemon = True)
        inputstream_thread.start()
        print ('Input stream thread started')
    if behavior == 'attraction':
        attraction_thread = threading.Thread(target=robot_move.attraction_only, daemon = True)
        attraction_thread.start()
    elif behavior == 'repulsion':
        repulsion_thread = threading.Thread(target=robot_move.repulsion_only, daemon = True)
        repulsion_thread.start()
    elif behavior == 'dynamic_movement':
        move_thread = threading.Thread(target=robot_move.audio_move, daemon = True)
        move_thread.start()
    else:
        print('No valid behavior provided')
    

    event = threading.Event() 
    
    time2 = startime.strftime('_%Y-%m-%d__%H-%M-%S')
    os.makedirs(folder_path, exist_ok=True)
    name = 'TIMESTAMPS_' + str(raspi_local_ip) + str(time2)
    npy_data = os.path.join(folder_path, name)

    # timestamp_queue.put([audio_processor.ts_queue.get(), 0, 'recording_start'])
    timestamp_queue.put([audio_processor.ts_queue.get(), 0, 0, 'recording_start'])
    try:
        while True:
            start_time = time.time()
            event.clear()
            timestamp = datetime.datetime.timestamp(datetime.datetime.now(datetime.timezone.utc))  # POSIX timestamp
            timestamp_queue.put([timestamp, 0, 0, 'start'])  # Put the timestamp in the queue (no block=False, keeps all values)
            # np.save(npy_data, np.array([timestamp, 0, 'start'], dtype=object))
            with sd.OutputStream(samplerate=fs,
                                blocksize=0, 
                                device=usb_fireface_index, 
                                channels=2,
                                callback=audio_processor.callback_out, finished_callback=event.set, latency='low') as out_stream:
                                    with out_stream:
                                        event.wait()
                                        
                                        # print('event time =', time.time() - start_time)
                                        if method == 'CC':
                                            time.sleep(input_buffer_time*2.3)
                                    

            # print('out time =', time.time() - start_time)  
            # time.sleep(0.25)                     
            start_time_1 = time.time()
            if method == 'DAS':
                # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]  # Format timestamp to milliseconds
                timestamp = datetime.datetime.timestamp(datetime.datetime.now(datetime.timezone.utc))# POSIX timestamp
                args.angle, dB_SPL_level = audio_processor.update_das()
                timestamp_queue.put([timestamp,dB_SPL_level[0],args.angle])  # Put the timestamp in the queue (no block=False, keeps all values)
                # np.save(npy_data, np.array([timestamp,dB_SPL_level[0],args.angle], dtype=object))                
                # print(time.time() - start_time_1, 'DAS time')
                # print(time.time(), 'end time')
                angle_queue.put(args.angle)
                level_queue.put(dB_SPL_level)
                print('Angle:', args.angle, 'dB SPL:', dB_SPL_level[0])

            elif method == 'CC':
                # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]  # Format timestamp to milliseconds
                timestamp = datetime.datetime.timestamp(datetime.datetime.now(datetime.timezone.utc))# POSIX timestamp

                args.angle, dB_SPL_level = audio_processor.update_das()
                timestamp_queue.put([timestamp,dB_SPL_level[0],args.angle])  # Put the timestamp in the queue (no block=False, keeps all values)
                np.save(npy_data, np.array([timestamp,dB_SPL_level[0],args.angle], dtype=object))                
                angle_queue.put(args.angle)
                level_queue.put(dB_SPL_level)

                if isinstance(args.angle, (int, float, np.number)):
                    if np.isnan(args.angle):
                        angle_queue.put(None)

            else:
                print('No valid method provided')
                robot_move.stop()


    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))
        robot_move.stop()
        # Collect all remaining items from the queue and append to existing npy data
        remaining_data = []
        while not timestamp_queue.empty():
            remaining_data.append(timestamp_queue.get())
        if remaining_data:
            # Load existing data if file exists
            if os.path.exists(npy_data + ".npy"):
                existing_data = np.load(npy_data + ".npy", allow_pickle=True)
                # Ensure existing_data is always a list of records
                if existing_data.ndim == 1 and isinstance(existing_data[0], (list, np.ndarray)):
                    combined_data = np.concatenate([existing_data, remaining_data], axis=0)
                else:
                    combined_data = np.array([existing_data] + remaining_data, dtype=object)
            else:
                combined_data = np.array(remaining_data, dtype=object)
            np.save(npy_data, combined_data)
        print(f"Matrix has been saved to npy at {npy_data}\n")
        # full_path = os.path.join(folder_path_data, args.timestamp_filename)
        # with open(full_path, "w", newline='') as file:
        #     writer = csv.writer(file)
        #     # Write all items from the queue to the CSV file
        #     while not timestamp_queue.empty():
        #         writer.writerow(timestamp_queue.get())
        # print(f"Matrix has been saved as csv to {folder_path_data}\n")
    except Exception as err:
        # Stop robot
        robot_move.running = False
        robot_move.stop()
        # Collect all remaining items from the queue and append to existing npy data
        remaining_data = []
        while not timestamp_queue.empty():
            remaining_data.append(timestamp_queue.get())
        if remaining_data:
            # Load existing data if file exists
            if os.path.exists(npy_data + ".npy"):
                existing_data = np.load(npy_data + ".npy", allow_pickle=True)
                # Ensure existing_data is always a list of records
                if existing_data.ndim == 1 and isinstance(existing_data[0], (list, np.ndarray)):
                    combined_data = np.concatenate([existing_data, remaining_data], axis=0)
                else:
                    combined_data = np.array([existing_data] + remaining_data, dtype=object)
            else:
                combined_data = np.array(remaining_data, dtype=object)
            np.save(npy_data, combined_data)
        print(f"Matrix has been saved to npy at {npy_data}\n")
        # full_path = os.path.join(folder_path_data, args.timestamp_filename)
        # with open(full_path, "w", newline='') as file:
        #     writer = csv.writer(file)
        #     # Write all items from the queue to the CSV file
        #     while not timestamp_queue.empty():
        #         writer.writerow(timestamp_queue.get())
        # print(f"Matrix has been saved as csv to {folder_path_data}\n")
    except KeyboardInterrupt:
        robot_move.running = False
        robot_move.stop()
        # Collect all remaining items from the queue and append to existing npy data
        remaining_data = []
        while not timestamp_queue.empty():
            remaining_data.append(timestamp_queue.get())
        if remaining_data:
            # Load existing data if file exists
            if os.path.exists(npy_data + ".npy"):
                existing_data = np.load(npy_data + ".npy", allow_pickle=True)
                # Ensure existing_data is always a list of records
                if existing_data.ndim == 1 and isinstance(existing_data[0], (list, np.ndarray)):
                    combined_data = np.concatenate([existing_data, remaining_data], axis=0)
                else:
                    combined_data = np.array([existing_data] + remaining_data, dtype=object)
            else:
                combined_data = np.array(remaining_data, dtype=object)
            np.save(npy_data, combined_data)
        print(f"Matrix has been saved to npy at {npy_data}\n")
        # full_path = os.path.join(folder_path_data, args.timestamp_filename)
        # with open(full_path, "w", newline='') as file:
        #     writer = csv.writer(file)
        #     # Write all items from the queue to the CSV file
        #     while not timestamp_queue.empty():
        #         writer.writerow(timestamp_queue.get())
        # print(f"Matrix has been saved as csv to {folder_path_data}\n")
        inputstream_thread.join()
        move_thread.join()
        attraction_thread.join()
        repulsion_thread.join()
        print('\nRecording finished: ' + repr(args.filename)) 
        parser.exit(0)  
    def handle_sigterm(signum, frame):
        robot_move.running = False
        robot_move.stop()
        print("Process terminated. Thymio stopped.")
        parser.exit(0)
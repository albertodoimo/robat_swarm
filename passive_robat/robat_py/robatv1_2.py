
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
import pyroomacoustics as pra
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

# Create queues for storing data
shared_audio_queue = queue.Queue()
recording_bool = False  # Set to True to record audio, False to just process audio without recording
print('imports done')

# Get the index of the USB card
usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)

ni.ifaddresses('wlan0')
raspi_local_ip = ni.ifaddresses('wlan0')[2][0]['addr']
print('raspi_local_ip =', raspi_local_ip)

# Parameters for the DOA algorithms
trigger_level =  55 # dB SPL
critical_level = 70 # dB SPL
c = 343   # speed of sound
fs = 48000

rec_samplerate = 48000
input_buffer_time = 0.1 # seconds
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

# Parameters for the PRA algorithm
echo = pra.linear_2D_array(center=[(channels-1)*mic_spacing//2,0], M=channels, phi=0, d=mic_spacing)

# Parameters for the DAS algorithm
theta_das = np.linspace(-90, 90, 61) # angles resolution for DAS spectrum
N_peaks = 2 # Number of peaks to detect in DAS spectrum

# Parameters for the chirp signal
duration_out = 10e-3  # Duration in seconds
silence_dur = 20 # [ms] can probably pushed to 20 
amplitude = 0.5 # Amplitude of the chirp

t = np.linspace(0, duration_out, int(fs*duration_out))
start_f, end_f = 24e3, 2e2
sweep = signal.chirp(t, start_f, t[-1], end_f)
sweep *= signal.windows.tukey(sweep.size, 0.2)
sweep *= 0.8

silence_samples = int(silence_dur * fs/1000)
silence_vec = np.zeros((silence_samples, ))
full_sig = np.concatenate((sweep, silence_vec))

stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])
data = amplitude * np.float32(stereo_sig)

out_blocksize = int(len(data))  # Length of the output signal
print('out_blocksize =', out_blocksize)

#plot and save data 
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
#highpass_freq, lowpass_freq = [auto_hipas_freq ,auto_lowpas_freq]
highpass_freq, lowpass_freq = [200 ,20e3]
freq_range = [start_f, end_f]

cutoff = auto_hipas_freq # [Hz] highpass filter cutoff frequency
sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')

sensitivity_path = 'passive_robat/robat_py/Knowles_SPH0645LM4H-B_sensitivity.csv'
sensitivity = pd.read_csv(sensitivity_path)

in_sig = np.zeros(block_size, dtype=np.float32)  # Initialize the buffer for the audio input stream
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
max_speed = 200 #to be verified 

# Straight speed
speed = 100 
#print('\nspeed = ',speed, '\n')

# Turning speed
prop_turn_speed = 50
turn_speed = 100 
waiturn = 200 #turning time ms

left_sensor_threshold = 100
right_sensor_threshold = 100	


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
    av_above_level = -100

    if recording_bool == True:
        # Create folder for saving recordings
        time1 = startime.strftime('%Y-%m-%d')
        time2 = startime.strftime('_%Y-%m-%d__%H-%M-%S')
        save_path = '/home/thymio/robat_py/robat_swarm/passive_robat/robat_py'
        folder_name = str(raspi_local_ip)
        folder_path = os.path.join(save_path,'recordings', folder_name, time1)
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

    # Create instances of the AudioProcessor and RobotMove classes
    audio_processor = AudioProcessor(fs, channels, block_size, data, args, trigger_level, critical_level, mic_spacing, ref, highpass_freq, lowpass_freq, theta_das, N_peaks,
                                      usb_fireface_index, args.subtype, interp_sensitivity, tgtmic_relevant_freqs, args.filename, args.rec_samplerate, sos, sweep)
    robot_move = RobotMove(speed, turn_speed, left_sensor_threshold, right_sensor_threshold, critical_level, trigger_level, ground_sensors_bool = True)
    
    # Create threads for the audio input and recording
    if recording_bool == True:
        inputstream_thread = threading.Thread(target=
            audio_processor.continuos_recording, daemon = True)
        inputstream_thread.start()
    else:
        inputstream_thread = threading.Thread(target=
            audio_processor.input_stream, daemon = True)
        inputstream_thread.start()

    move_thread = threading.Thread(target=robot_move.audio_move, daemon = True)
    move_thread.start()
    event = threading.Event()

    try:
        while True:
            event.clear()
            start_time = time.time()      
            with sd.OutputStream(samplerate=fs,
                                blocksize=0, 
                                device=usb_fireface_index, 
                                channels=2,
                                callback=audio_processor.callback_out, finished_callback=event.set, latency='low') as out_stream:
                                    with out_stream:
                                        event.wait()
                                        print('event time =', time.time() - start_time)
                                        if method == 'CC':
                                            time.sleep(input_buffer_time*2.3)
            print('out time =', time.time() - start_time)                       
            start_time_1 = time.time()
              # Allow some time for the audio input to be processed
            if method == 'CC':
                  # Adjust this sleep time as needed
                # print('0 time =', time.time() - start_time_1) 
                args.angle, dB_SPL_level = audio_processor.update()  
                angle_queue.put(args.angle)
                level_queue.put(dB_SPL_level)


                if isinstance(args.angle, (int, float, np.number)):
                    if np.isnan(args.angle):
                        angle_queue.put(None)

            elif method == 'DAS':

                args.angle, dB_SPL_level = audio_processor.update_das()
                angle_queue.put(args.angle)
                level_queue.put(dB_SPL_level)

            else:
                print('No valid method provided')
                robot_move.stop()


            print('in time =', time.time() - start_time_1)
        else:
            robot_move.stop()

    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))
        robot_move.stop()
    except Exception as err:
        # Stop robot
        robot_move.stop()
        print('err:',err)
    except KeyboardInterrupt:
        robot_move.running = False
        robot_move.stop()
        inputstream_thread.join()
        move_thread.join()
        print('\nRecording finished: ' + repr(args.filename)) 
        parser.exit(0)  
        
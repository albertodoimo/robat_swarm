
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

from scipy.ndimage import gaussian_filter1d
from thymiodirect import Connection 
from thymiodirect import Thymio
from functions.das_v2 import das_filter_v2
from functions.music import music
from functions.get_card import get_card 
from functions.pow_two_pad_and_window import pow_two_pad_and_window
from functions.check_if_above_level import check_if_above_level
from functions.calc_multich_delays import calc_multich_delays
from functions.avar_angle import avar_angle
from functions.bandpass import bandpass
from functions.save_data_to_csv import save_data_to_csv
from functions.save_data_to_xml import save_data_to_xml
from RobotMove import RobotMove
from shared_queues import angle_queue, level_queue

print('imports done')

# Get the index of the USB card
usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)

# Parameters for the DOA algorithms
trigger_level = -60 # dB level ref max pdm
critical_level = -40 # dB level pdm critical distance
c = 343   # speed of sound
fs = 48000
rec_samplerate = 48000
block_size = 512 #used for the shared queue from which the doa is computed, not anymore for the output stream
channels = 5
mic_spacing = 0.018 #m
nfft = 512
ref = channels//2 #central mic in odd array as ref
#print('ref=',ref) 
#ref= 0 #left most mic as reference
critical = []
# print('critical', np.size(critical))

# Possible algorithms for computing DOA:CC, DAS
method = 'CC'

# Parameters for the CC algorithm
avar_theta = None
theta_values   = []

# Parameters for the PRA algorithm
echo = pra.linear_2D_array(center=[(channels-1)*mic_spacing//2,0], M=channels, phi=0, d=mic_spacing)

# Parameters for the DAS algorithm
theta_das = np.linspace(-90, 90, 61) # angles resolution for DAS spectrum
N_peaks = 1 # Number of peaks to detect in DAS spectrum

# Parameters for the chirp signal
rand = random.uniform(0.8, 1.2)
duration_out = 20e-3  # Duration in seconds
silence_dur = 60 # [ms]
amplitude = 0 # Amplitude of the chirp

# Generate a chirp signal
low_freq = 1e3 # [Hz]
hi_freq =  20e3 # [Hz]
t_tone = np.linspace(0, duration_out, int(fs*duration_out))
chirp = signal.chirp(t_tone, low_freq, t_tone[-1], hi_freq)
sig = pow_two_pad_and_window(chirp, fs = fs, show=False)

silence_samples = int(silence_dur * fs/1000)
silence_vec = np.zeros((silence_samples, ))
full_sig = np.concatenate((sig, silence_vec))
print('len = ', len(full_sig))
stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])
data = amplitude * np.float32(stereo_sig)

# Calculate highpass and lowpass frequencies based on the array geometry
auto_hipas_freq = int(343/(2*(mic_spacing*(channels-1))))
print('HP frequency:', auto_hipas_freq)
auto_lowpas_freq = int(343/(2*mic_spacing))
print('LP frequency:', auto_lowpas_freq)
#highpass_freq, lowpass_freq = [auto_hipas_freq ,auto_lowpas_freq]
highpass_freq, lowpass_freq = [20 ,20e3]
freq_range = [hi_freq, low_freq]

# Thymio movement parameters
max_speed = 200 #to be verified 

# Straight speed
speed = 150 
#print('\nspeed = ',speed, '\n')

# Turning speed
prop_turn_speed = 50
turn_speed = 100 
waiturn = 200 #turning time ms

left_sensor_threshold = 100
right_sensor_threshold = 100	

# Create queues for storing data
shared_audio_queue = queue.Queue()

# Stream callback function
class AudioProcessor:
    def __init__(self, fs, channels, block_size, data, args, trigger_level, critical_level, mic_spacing, ref, highpass_freq, lowpass_freq, theta_das, N_peaks):
        self.fs = fs
        self.channels = channels
        self.block_size = block_size
        self.data = data
        self.args = args
        self.trigger_level = trigger_level
        self.critical_level = critical_level
        self.mic_spacing = mic_spacing
        self.ref = ref
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        self.theta_das = theta_das
        self.N_peaks = N_peaks
        self.q = queue.Queue()
        self.qq = queue.Queue()
        self.qqq = queue.Queue()
        self.shared_audio_queue = queue.Queue()
        self.current_frame = 0

    def continuos_recording(self):
        with sf.SoundFile(args.filename, mode='x', samplerate=args.rec_samplerate,
                            channels=args.channels, subtype=args.subtype) as file:
            with sd.InputStream(samplerate=fs, device=usb_fireface_index,channels=channels, callback=audio_processor.callback_in, blocksize=self.block_size):
                    while True:
                        file.write(self.shared_audio_queue.get())
        
    def callback_out(self, outdata, frames, time, status):
        if status:
            print(status)
        chunksize = min(len(self.data) - self.current_frame, frames)
        outdata[:chunksize] = self.data[self.current_frame:self.current_frame + chunksize]
        if chunksize < frames:
            outdata[chunksize:] = 0
            self.current_frame = 0  # Reset current_frame after each iteration
            raise sd.CallbackStop()
        self.current_frame += chunksize
            
    def callback_in(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        self.shared_audio_queue.put((indata).copy())
        
#    def callback(self, indata, outdata, frames, time, status):
#        if status:
#            print(status)
#        chunksize = min(len(self.data) - self.current_frame, frames)
#        outdata[:chunksize] = self.data[self.current_frame:self.current_frame + chunksize]
#        if chunksize < frames:
#            outdata[chunksize:] = 0
#            self.current_frame = 0  # Reset current_frame after each iteration
#            raise sd.CallbackStop()
#        self.current_frame += chunksize
#        self.q.put((indata).copy())
#        self.args.buffer = ((indata).copy())

    def update(self):
        in_buffer = self.shared_audio_queue.get()
        in_sig = in_buffer-np.mean(in_buffer)

        #ref_channels_bp = bandpass(in_sig,self.highpass_freq,self.lowpass_freq,self.fs)
        #print('ref_channels_bp=', np.shape(ref_channels_bp))
        trigger_bool,critical_bool,dBrms_channel = check_if_above_level(in_sig,self.trigger_level,self.critical_level)
        print(trigger_bool)
        print(critical_bool)
        max_dBrms = np.max(dBrms_channel)
        print('max dBrms =',max_dBrms)
        av_above_level = np.mean(dBrms_channel)
        #print(av_above_level)
        ref_sig = in_sig[:,self.ref]
        delay_crossch= calc_multich_delays(in_sig,ref_sig,self.fs,self.ref)

        # calculate avarage angle
        avar_theta = avar_angle(delay_crossch,self.channels,self.mic_spacing,self.ref)
        
        time3 = datetime.datetime.now()
        avar_theta1 = np.array([avar_theta, time3.strftime('%H:%M:%S.%f')[:-3]])

        #print('avarage theta',avar_theta1)
            
        if trigger_bool or critical_bool:
            #print('avarage theta deg = ', np.rad2deg(avar_theta))
            return np.rad2deg(avar_theta), max_dBrms
        else:
            avar_theta = None
            return avar_theta, max_dBrms

    def update_das(self):
        # Update  with the DAS algorithm
        in_buffer = self.shared_audio_queue.get()
        in_sig = in_buffer-np.mean(in_buffer)

        #print('ref_channels_bp=', np.shape(ref_channels_bp))
        trigger_bool,critical_bool,dBrms_channel = check_if_above_level(in_sig,self.trigger_level,self.critical_level)
        print(trigger_bool)
        print(critical_bool)
        max_dBrms = np.max(dBrms_channel)
        print('max dBrms =',max_dBrms)

        #print('buffer', np.shape(in_sig))
        #starttime = time.time()
        theta, spatial_resp = das_filter_v2(in_sig, self.fs, self.channels, self.mic_spacing, [self.highpass_freq, self.lowpass_freq], theta=self.theta_das)
        #theta, spatial_resp = music(in_sig, self.fs, self.channels, self.mic_spacing, [self.highpass_freq, self.lowpass_freq], theta=self.theta_das, show = True)
        
        #print(time.time()-starttime)
        # find the spectrum peaks 

        spatial_resp = gaussian_filter1d(spatial_resp, sigma=4)
        peaks, _ = signal.find_peaks(spatial_resp)  # Adjust height as needed
        # peak_angle = theta_das[np.argmax(spatial_resp)]
        peak_angles = theta[peaks]
        N = self.N_peaks # Number of peaks to keep

        # Sort peaks by their height and keep the N largest ones
        peak_heights = spatial_resp[peaks]
        top_n_peak_indices = np.argsort(peak_heights)[-N:]  # Indices of the N largest peaks # Indices of the N largest peaks
        top_n_peak_indices = top_n_peak_indices[::-1]
        peak_angles = theta[peaks[top_n_peak_indices]]  # Corresponding angles
        print('peak angles', peak_angles)

        if trigger_bool or critical_bool:
            #print('avarage theta deg = ', np.rad2deg(avar_theta))
            return peak_angles, max_dBrms
        else:
            peak_angles = None
            return peak_angles, max_dBrms

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

    # Create folder for saving recordings
    time1 = startime.strftime('%Y-%m-%d__%H-%M-%S')
    save_path = '/home/thymio/robat_py/'
    folder_name = str(time1)  
    folder_path = os.path.join(save_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    name = 'MULTIWAV_' + str(time1) + '.wav'
    args.filename = os.path.join(folder_path, name)

    if args.samplerate is None:  
        print('error!: no samplerate set! Using default')
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = int(device_info['default_samplerate'])
    if args.filename is None:
        timenow = datetime.datetime.now()
        time1 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
        args.filename = 'MULTIWAV_' + str(time1) + '.wav'
    print(args.samplerate)

    # Create instances of the AudioProcessor and RobotMove classes
    audio_processor = AudioProcessor(fs, channels, block_size, data, args, trigger_level, critical_level, mic_spacing, ref, highpass_freq, lowpass_freq, theta_das, N_peaks)
    robot_move = RobotMove(speed, turn_speed, left_sensor_threshold, right_sensor_threshold, critical_level, trigger_level, ground_sensors_bool = True)
    
    # Create threads for the audio input and recording
    inputstream_thread = threading.Thread(target=
        audio_processor.continuos_recording, daemon = True)
    inputstream_thread.start()

    #angle_queue.put(None)
    move_thread = threading.Thread(target=robot_move.audio_move, daemon = True)
    move_thread.start()

    try:
        #audio_processor.continuos_recording()
        while True:
            current_frame = 0 
            start_time = time.time()      
            with sd.OutputStream(samplerate=fs,
                                blocksize=0, 
                                device=usb_fireface_index, 
                                channels=2,
                                callback=audio_processor.callback_out,
                                latency='low') as out_stream:
                                while out_stream.active:
                                    #robot_move.stop()
                                    pass
            #print('out time =', time.time() - start_time)                       
            #start_time = time.time()
            
            if method == 'CC':
                args.angle, max_dBrms = audio_processor.update()
                angle_queue.put(args.angle)
                level_queue.put(max_dBrms)

                if isinstance(args.angle, (int, float, np.number)):
                    if np.isnan(args.angle):
                        angle_queue.put(None)

            elif method == 'DAS':

                args.angle, max_dBrms = audio_processor.update_das()
                angle_queue.put(args.angle)
                level_queue.put(max_dBrms)

            else:
                print('No valid method provided')


            time_start_robot = time.time()
            #print('in time =', time.time() - time_start)
        else:
            #print('in time =', time.time() - start_time)
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
        # Stop robot
        

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

print('imports done')

# Get the index of the USB card
usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)

# Parameters for the DOA algorithms
trigger_level = -60 # dB level ref max pdm
critical_level = -30 # dB level pdm critical distance
c = 343   # speed of sound
fs = 48000
rec_samplerate = 44000
block_size = 512
analyzed_buffer = fs/block_size #theoretical buffers analyzed each second 
channels = 5
mic_spacing = 0.018 #m
nfft = 512
ref = channels//2 #central mic in odd array as ref
#print('ref=',ref) 
#ref= 0 #left most mic as reference
critical = []
# print('critical', np.size(critical))

# Possible algorithms for computing DOA: PRA (pyroomacoustics), CC, DAS
method = 'CC'

doa_name = 'MUSIC'
doa_name = 'SRP'

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
duration_in = rand * 0.5  # Duration in seconds
duration_in = 500e-3  # Duration in seconds
amplitude = 0.1 # Amplitude of the chirp

# Generate a chirp signal
low_freq = 1e3 # [Hz]
hi_freq =  20e3 # [Hz]
t_tone = np.linspace(0, duration_out, int(fs*duration_out))
chirp = signal.chirp(t_tone, low_freq, t_tone[-1], hi_freq)
sig = pow_two_pad_and_window(chirp, fs = fs, show=False)
silence_dur = 100 # [ms]
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
highpass_freq, lowpass_freq = [auto_hipas_freq ,auto_lowpas_freq]
freq_range = [highpass_freq, lowpass_freq]
freq_range = [hi_freq, low_freq]

# Create queues for storing data
q = queue.Queue()
qq = queue.Queue()
qqq = queue.Queue()

# Stream callback function
current_frame = 0
def callback_out(outdata, frames, time, status):
    global current_frame
    if status:
        print(status)
    chunksize = min(len(data) - current_frame, frames)
    outdata[:chunksize] = data[current_frame:current_frame + chunksize]
    if chunksize < frames:
        outdata[chunksize:] = 0
        current_frame = 0  # Reset current_frame after each iteration
        raise sd.CallbackStop()
    current_frame += chunksize
        
def callback_in(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    #correction=np.mean(indata)
    #print(correction)
    #in_sig = indata-correction
    q.put((indata).copy())
    args.buffer = ((indata).copy())
    
def callback(indata, outdata, frames, time, status):
    global current_frame
    if status:
        print(status)
    chunksize = min(len(data) - current_frame, frames)
    outdata[:chunksize] = data[current_frame:current_frame + chunksize]
    if chunksize < frames:
        outdata[chunksize:] = 0
        current_frame = 0  # Reset current_frame after each iteration
        raise sd.CallbackStop()
    current_frame += chunksize
    q.put((indata).copy())
    args.buffer = ((indata).copy())

def record(rec_counter,time1):
    q_contents = [q.get() for _ in range(q.qsize())]

    #print('q_contents = ', np.shape(q_contents))
    rec = np.concatenate(q_contents)
    #print('rec = ', np.shape(rec))


    rec2besaved = rec[:, :channels]
    save_path = '/home/thymio/robat_py/robat_v0_files/'
    #save_path = ''
    # Create folder with args.filename (without extension)
    folder_name = str(time1) + '_MULTIWAV'  
    folder_path = os.path.join(save_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    save_path = folder_path

    timenow = datetime.datetime.now()
    time2 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
    full_path = os.path.join(save_path, str(rec_counter)+'__'+time2+'.wav')
    with sf.SoundFile(full_path, mode='x', samplerate=rec_samplerate,
                channels=args.channels, subtype=args.subtype) as file:\
        file.write(rec2besaved)
    print(f'\nsaved to {full_path}\n')

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def update():

    correction=np.mean(args.buffer)
    in_sig = args.buffer-correction
    ref_channels = in_sig
    #print(np.shape(in_sig))
    #print(in_sig)
    #print('ref_channels=', np.shape(ref_channels))
    ref_channels_bp = bandpass(ref_channels,highpass_freq,lowpass_freq,fs)
    #print('ref_channels_bp=', np.shape(ref_channels_bp))
    above_level,dBrms_channel = check_if_above_level(ref_channels_bp,trigger_level)
    #print(above_level)
    av_above_level = np.mean(dBrms_channel)
    #print(av_above_level)
    ref_sig = in_sig[:,ref]
    delay_crossch= calc_multich_delays(in_sig,ref_sig,fs,ref)

    # calculate avarage angle
    avar_theta = avar_angle(delay_crossch,channels,mic_spacing,ref)
    time3 = datetime.datetime.now()
    avar_theta1 = np.array([avar_theta, time3.strftime('%H:%M:%S.%f')[:-3]])

    #print('avarage theta',avar_theta1)

    qqq.put(avar_theta1.copy()) # store detected angle value
    #if av_above_level > critical_level:
    #    avar_theta = critical
    #    return avar_theta
        
    if av_above_level > trigger_level:


        #calculate avarage angle
        #avar_theta = avar_angle(delay_crossch,channels,mic_spacing,ref)
        #avar_theta = avar_angle(delay_crossch_gcc,channels,mic_spacing,ref)
        
        print('avarage theta deg = ', np.rad2deg(avar_theta))
        return np.rad2deg(avar_theta), av_above_level
    else:
        avar_theta = None
        return avar_theta, av_above_level

def update_polar():
    # Your streaming data source logic goes here

    in_sig = args.buffer
    #print('buffer', np.shape(in_sig))

    X = pra.transform.stft.analysis(in_sig, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])

    doa = pra.doa.algorithms[doa_name](echo, fs, nfft, c=c, num_src=2, max_four=2)
    doa.locate_sources(X, freq_range=freq_range)
    #print('azimuth_recon=',doa.azimuth_recon) # rad value of detected angles
    theta_pra_deg = (doa.azimuth_recon * 180 / np.pi) 
    #print('theta=',theta_pra_deg) #degrees value of detected angles

    spatial_resp = doa.grid.values # 360 values for plot
    #print('spat_resp',spatial_resp) 

    # normalize   
    min_val = spatial_resp.min()
    max_val = spatial_resp.max()
    spatial_resp = (spatial_resp - min_val) / (max_val - min_val)
    time3 = datetime.datetime.now()

    spatial_resp = np.hstack([spatial_resp, np.array(time3.strftime('%H:%M:%S.%f')[:-3])])
    

    qq.put(spatial_resp.copy())
    qqq.put(doa.azimuth_recon.copy())

    ref_channels = in_sig
    #print('ref_channels=', np.shape(ref_channels))
    ref_channels_bp = bandpass(ref_channels,highpass_freq,lowpass_freq,fs)
    #print('ref_channels_bp=', np.shape(ref_channels_bp))
    above_level,dBrms_channel = check_if_above_level(ref_channels_bp,trigger_level)
    #print(above_level)
    av_above_level = np.mean(dBrms_channel)
    #print(av_above_level)

    if av_above_level > critical_level:
        theta_pra = critical
        print('critical')
        return theta_pra

    elif av_above_level > trigger_level:

        if theta_pra_deg[0]<=180:
            theta_pra = 90-theta_pra_deg[0]
            print('pra theta deg 0', theta_pra)
            return theta_pra
        elif theta_pra_deg[1]<=180:
            theta_pra = 90-theta_pra_deg[1]
            print('pra theta deg 1', theta_pra)
            return theta_pra
        else:
            theta_pra = None
            return theta_pra

def update_das():
    # Update  with the DAS algorithm

    correction=np.mean(args.buffer)
    in_sig = args.buffer-correction

    #print('buffer', np.shape(in_sig))
    #starttime = time.time()
    #theta, spatial_resp = das_filter_v2(in_sig, fs, channels, mic_spacing, freq_range, theta=theta_das)
    theta, spatial_resp = music(in_sig, fs, channels, mic_spacing, freq_range, theta=theta_das, show = False)
    
    #print(time.time()-starttime)
    # find the spectrum peaks 

    spatial_resp = gaussian_filter1d(spatial_resp, sigma=4)
    peaks, _ = signal.find_peaks(spatial_resp)  # Adjust height as needed
    # peak_angle = theta_das[np.argmax(spatial_resp)]
    peak_angles = theta[peaks]
    N = N_peaks # Number of peaks to keep

    # Sort peaks by their height and keep the N largest ones
    peak_heights = spatial_resp[peaks]
    top_n_peak_indices = np.argsort(peak_heights)[-N:]  # Indices of the N largest peaks # Indices of the N largest peaks
    top_n_peak_indices = top_n_peak_indices[::-1]
    peak_angles = theta[peaks[top_n_peak_indices]]  # Corresponding angles
    print('peak angles', peak_angles)

    return peak_angles
    print('Angle = ', peak_angles)
    #qq.put(spatial_resp.copy())

def main(use_sim=False, ip='localhost', port=2001):
    ''' Main function '''
    try:
        global startime
        # Configure Interface to Thymio robot
        # simulation
        port = Connection.serial_default_port()
        th = Thymio(serial_port=port, 
                    on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
        # Connect to Robot
        th.connect()
        robot = th[th.first_node()]

        startime = datetime.datetime.now()
        args.samplerate = fs
        if args.samplerate is None:  
            print('error!: no samplerate set! Using default')
            device_info = sd.query_devices(args.device, 'input')
            args.samplerate = int(device_info['default_samplerate'])
        if args.filename is None:
            timenow = datetime.datetime.now()
            time1 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
            args.filename = 'MULTIWAV_' + str(time1) + '.wav'
        print(args.samplerate)

        while True: 
            current_frame = 0       
            with sd.OutputStream(samplerate=fs,
                                blocksize=0, 
                                device=usb_fireface_index, 
                                channels=2,
                                callback=callback_out,
                                latency='low') as out_stream:
                                while out_stream.active:
                                    robot['motor.left.target'] = 0
                                    robot['motor.right.target'] = 0
            start_time = time.time()

            with sd.InputStream(samplerate=fs, device=usb_fireface_index,channels=channels, callback=callback_in, blocksize=block_size) as input_stream:
                while time.time() - start_time < duration_in:
                    #print('in time =', time.time() - start_time)

#            while True:
#                if (time.time() - start_time_rec) <= 45: #seconds
#                    pause = False
#                    pass
#                else:
#                    pause = True
#                    #record(rec_counter,time1)
#
#                    q_contents = [q.get() for _ in range(q.qsize())]
#
#                    
#                    #time.sleep(1)
#                    print('q_contents = ', np.shape(q_contents))
#                    rec = np.concatenate(q_contents)
#                    print('rec = ', np.shape(rec))
#                    
#
#                    rec2besaved = rec[:, :channels]
#                    save_path = '/home/thymio/robat_py/'
#                    save_path = ''
#                    # Create folder with args.filename (without extension)
#                    folder_name = str(time1) + '_MULTIWAV'  
#                    folder_path = os.path.join(save_path, folder_name)
#                    os.makedirs(folder_path, exist_ok=True)
#                    save_path = folder_path
#
#                    timenow = datetime.datetime.now()
#                    time2 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
#                    full_path = os.path.join(save_path, str(rec_counter)+'.wav')
#                    with sf.SoundFile(full_path, mode='x', samplerate=rec_samplerate,
#                                    channels=args.channels, subtype=args.subtype) as file:
#                        file.write(rec2besaved)
#                        print(f'\nsaved to {full_path}\n')
#                    rec_counter += 1
#                    start_time_rec = time.time()
#                    record(rec_counter,time1)
#                    # print('delta save=',time.time()-start_time_rec,'sec')
#                    #print('startime = ',start_time_rec)

                    # Check if the elapsed time since the last frame is less than or equal to the desired frame duration

                    if method == 'PRA':
                        time_start = time.time()
                        angle = update_polar()
                        #print('Angle = ', angle)
                        time_end = time.time()
                        #print('delta update pra',time_end - time_start,'sec')

                    elif method == 'CC':
                        time_start = time.time()
                        angle, av_above_level = update()
                        if isinstance(angle, (int, float, np.number)):
                            if np.isnan(angle):
                                angle = None
                        #else:
                        #    print("Warning: angle is not a numerical type")
                            
                        #print('Angle = ', angle)
                        #print('av ab level = ', av_above_level)
                        #angle = int(angle)
                        
                        #print('updatestart=',datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S'))
                        time_end = time.time()
                        #print('delta update cc',time_end - time_start,'sec')

                    elif method == 'DAS':

                        time_start = time.time()

                        correction=np.mean(args.buffer)
                        in_sig = args.buffer-correction
                        ref_channels = in_sig
                        #print('ref_channels=', np.shape(ref_channels))
                        ref_channels_bp = bandpass(ref_channels,highpass_freq,lowpass_freq,fs)
                        #print('ref_channels_bp=', np.shape(ref_channels_bp))
                        above_level, dBrms_channel = check_if_above_level(ref_channels_bp,trigger_level)
                        #print(above_level)
                        av_above_level = np.mean(dBrms_channel)
                        #print(av_above_level)

                        if av_above_level > critical_level:
                            print('critical')
                            angle = update_das()

                        elif av_above_level > trigger_level:
                            angle = update_das()
                        else:
                            angle = None


                        #print('Angle = ', angle)
                        #print('av ab level = ', av_above_level)
                        #angle = int(angle)
                        
                        #print('updatestart=',datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S'))
                        time_end = time.time()
                        #print('delta update das',time_end - time_start,'sec')
                        start_time_rec = time.time()

                    
                    else:
                        print('No valid method provided')

                    time_start_robot = time.time()
                    ground_sensors = robot['prox.ground.reflected']
                    #print('ground = ',robot['prox.ground.reflected'])
                    
                    # Adjust these threshold values as needed
                    left_sensor_threshold = 300
                    right_sensor_threshold = 300	

                    waiturn = 0.05
                    norm_coefficient = 48000/1024 #most used fractional value, i.e. normalization value 
                    max_speed = 100 #to be verified 
                    speed = 0.5 * max_speed * analyzed_buffer *(1/norm_coefficient) #generic speed of robot while moving 
                    speed = int(speed)
                    #print('\nspeed = ',speed, '\n')
                    if angle != None:
                        turn_speed = 0.5 * 1/90 * (speed*int(angle)) #velocity of the wheels while turning 
                        turn_speed = int(turn_speed)

                        print('\nturn_speed = ',turn_speed, '\n')
                    direction = random.choice(['left', 'right'])
                    turn_speed = 500
                        #PROPORTIONAL MOVEMENT
                    if ground_sensors[0] > left_sensor_threshold  and ground_sensors[1]> right_sensor_threshold:
                        # Both sensors detect the line, turn left
                        if direction == 'left':
                            robot['motor.left.target'] = -speed
                            robot['motor.right.target'] = speed   
                            time.sleep(waiturn) 
                            pass
                        else:
                            robot['motor.left.target'] = speed
                            robot['motor.right.target'] = -speed
                            time.sleep(waiturn)
                            pass
                    elif ground_sensors[1] > right_sensor_threshold:
                        # Only right sensor detects the line, turn left
                        robot['motor.left.target'] = -speed
                        robot['motor.right.target'] = speed
                        time.sleep(waiturn)
                    elif ground_sensors[0] > left_sensor_threshold:
                        # Only left sensor detects the line, turn right
                        robot['motor.left.target'] = speed 
                        robot['motor.right.target'] = -speed 
                        time.sleep(waiturn) 
                    
                    else:  #attraction or repulsion 
                        if angle == None: #neutral movement
                            robot["leds.top"] = [0, 0, 0]
                            robot['motor.left.target'] = speed
                            robot['motor.right.target'] = speed
                            
                        elif av_above_level > critical_level: #repulsion
                            robot['motor.left.target'] =  - turn_speed
                            robot['motor.right.target'] =  turn_speed
                            time.sleep(waiturn)
                        else: #attraction
                            if angle < 0:
                                robot['motor.left.target'] =   turn_speed
                                robot['motor.right.target'] =  - turn_speed
                                time.sleep(waiturn)
                            else:
                                robot['motor.left.target'] =   turn_speed
                                robot['motor.right.target'] = - turn_speed
                                time.sleep(waiturn)

                        #print('delta robot',time.time() - time_start_robot,'sec')
                else:
                    robot['motor.left.target'] = 0
                    robot['motor.right.target'] = 0

    except Exception as err:
        # Stop robot
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0 
        robot["leds.top"] = [0,0,0]
        print('err:',err)
    except KeyboardInterrupt:
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0
        robot["leds.top"] = [0,0,0]
        print("Press Ctrl-C again to end the program")    

#        save_path = '/home/thymio/robat_py/robat_v0_files/'
#        #save_path = ''
#        folder_name = str(time1) + '_rec_data' 
#        folder_path = os.path.join(save_path, folder_name)
#        os.makedirs(folder_path, exist_ok=True)

#        if method == 'CC':
#            theta_doa = np.vstack([[qqq.get() for _ in range(qqq.qsize())]]) #recognized angle values
#            file_theta_doa_xml = time1 + "_theta_doa_CC.xml"
#            file_theta_doa_csv = time1 + "_theta_doa_CC.csv"
#            print(np.shape(theta_doa))
#            print(theta_doa)
#            #save_data_to_xml(theta_doa, file_theta_doa_xml, folder_path) #[time, av angle deg]
#            #save_data_to_csv(theta_doa, file_theta_doa_csv, folder_path) #[time, av angle deg]
#            
#        if method == 'PRA':
#            spatial_response = [qq.get() for _ in range(qq.qsize())] #360 values for plot
#            print(np.shape(spatial_response))
#
#            theta_doa = [[qqq.get() for _ in range(qqq.qsize())]] #recognized angle values
#
#            file_spat_resp_xml = time1 + "_spat_resp_PRA.xml" 
#            file_spat_resp_csv =  time1 + "_spat_resp_PRA.csv" 
#            #save_data_to_xml(spatial_response, file_spat_resp_xml, folder_path) #[time, 360 polar plot angles]
#            #save_data_to_csv(spatial_response, file_spat_resp_csv, folder_path) #[time, 360 polar plot angles]
#            
#            file_theta_doa_xml = time1 + "_theta_doa_PRA.xml" #[time, number of detected angle]
#            file_theta_doa_csv = time1 + "_theta_doa_PRA.csv" #[time, number of detected angle]
#            #save_data_to_xml(theta_doa, file_theta_doa_xml, folder_path)
#            #save_data_to_csv(theta_doa, file_theta_doa_csv, folder_path)

        
        # save audio
        q_contents = [q.get() for _ in range(q.qsize())]
        #print('q_contents = ', np.shape(q_contents))
        rec = np.concatenate(q_contents)
        #print('rec = ', np.shape(rec))
        
#        rec2besaved = rec[:, :channels]
#
#        folder_name = str(time1) + '_MULTIWAV'
#        folder_path = os.path.join(save_path, folder_name)
#        os.makedirs(folder_path, exist_ok=True)
#        save_path = folder_path
#
#        full_path = os.path.join(save_path, 'end.wav')
#        with sf.SoundFile(full_path, mode ='x', samplerate=rec_samplerate,
#                        channels=args.channels, subtype=args.subtype) as file:
#            file.write(rec2besaved)
#            print(f'\nsaved to {args.filename}\n')
        

if __name__ == '__main__':
    # Parse commandline arguments to cofigure the interface for a simulation (default = real robot)
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                    'If no arguments are given, the code will run with a real Thymio.', add_help=False)
    
    # Add optional arguments
    parser.add_argument('-s', '--sim', action='store_true', help='set this flag to use simulation')
    parser.add_argument('-i', '--ip', help='set the TCP host ip for simulation. default=localhost', default='localhost')
    parser.add_argument('-p', '--port', type=int, help='set the TCP port for simulation. default=2001', default=2001)


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
    main(args.sim, args.ip, args.port)

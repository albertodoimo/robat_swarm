
# Robat v0 code to use with different arrays(PDM passive array or I2S array).
#
# FEATURES:
# > Compute pra alg for angle detection
# > Calculates dB values 
# > Triggers computation of angle only over threshold
# > Save recordings every x minutes 
# > Save file data from angle/polar plot in csv and xml files

print('import libraries...')

import numpy as np
import matplotlib.pyplot as plt
import glob
import pyroomacoustics as pra
import scipy.signal as signal 
import sounddevice as sd
import soundfile as sf

from matplotlib.animation import FuncAnimation
import argparse
import queue
import sys
import datetime
import time
import math
import random
import os
import csv
import xml.etree.ElementTree as ET
from scipy.fftpack import fft, ifft

from thymiodirect import Connection 
from thymiodirect import Thymio
from scipy import signal

print('libraries imported')

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)


# Possible algorithms for computing DOA: PRA (pyroomacoustics), CC
method = 'CC'
#method = 'PRA'
doa_name = 'MUSIC'
doa_name = 'SRP'

c = 343   # speed of sound
fs = 44000
rec_samplerate = 44000
block_size = 1024
analyzed_buffer = fs/block_size #theoretical buffers analyzed each second 
channels = 5
mic_spacing = 0.018 #m
ref = channels//2 #central mic in odd array as ref
print('ref=',ref) 
#ref= 0 #left most mic as reference
nfft = 32  # FFT size

#fps = 25
#duration = 60 #seconds

auto_hipas_freq = int(343/(2*(mic_spacing*(channels-1))))
print('HP frequency:', auto_hipas_freq)
auto_lowpas_freq = int(343/(2*mic_spacing))
print('LP frequency:', auto_lowpas_freq)

highpass_freq, lowpass_freq = [20 ,auto_lowpas_freq]
freq_range = [highpass_freq, lowpass_freq]

nyq_freq = fs/2.0
b, a = signal.butter(4, [highpass_freq/nyq_freq,lowpass_freq/nyq_freq],btype='bandpass') # to be 'allowed' in Hz.

#detect_trig_lev = True
#
#if detect_trig_lev:
#    S = sd.InputStream(samplerate=fs, device=usb_fireface_index,
#                            channels=channels, blocksize=block_size)
#    S.start()
#    ref_channels, status = S.read(S.blocksize)
#    squared = np.square(ref_channels)
#    mean_squared = np.mean(squared)
#    root_mean_squared = np.sqrt(mean_squared)
#    dBrms_channel = 20.0*np.log10(root_mean_squared)
#    av_above_level = np.mean(dBrms_channel)
#    trigger_level = av_above_level
#    print('trigger_level = ', av_above_level)
#    S.stop()
#    sd.stop()
#    time.sleep(1)
#else: 
#    #trigger_level = -25.2 # dB level ref max 12s
trigger_level = -54 # dB level ref max pdm
critical_level = -48 # dB level pdm critical distance
critical = []
print('critical', np.size(critical))
echo = pra.linear_2D_array(center=[(channels-1)*mic_spacing//2,0], M=channels, phi=0, d=mic_spacing)

print('loading functions...')

q = queue.Queue()
qq = queue.Queue()
qqq = queue.Queue()

def bandpass_sound(rec_buffer,a,b):
    """
    """
    rec_buffer_bp = np.apply_along_axis(lambda X : signal.lfilter(b, a, X),0, rec_buffer)
    return(rec_buffer_bp)

def check_if_above_level(mic_inputs,trigger_level):
    """Checks if the dB rms level of the input recording buffer is above
    threshold. If any of the microphones are above the given level then 
    recording is initiated. 
    
    Inputs:
        
        mic_inputs : Nsamples x Nchannels np.array. Data from soundcard
        
        level : integer <=0. dB rms ref max . If the input data buffer has an
                dB rms >= this value then True will be returned. 
                
    Returns:
        
        above_level : Boolean. True if the buffer dB rms is >= the trigger_level
    """ 

    dBrms_channel = np.apply_along_axis(calc_dBrms, 0, mic_inputs)   
    #print('dBrms_channel=',dBrms_channel)     
    above_level = np.any( dBrms_channel >= trigger_level)
    #print('above level =',above_level)
    return(above_level,dBrms_channel)

def calc_dBrms(one_channel_buffer):
    """
    """
    squared = np.square(one_channel_buffer)
    mean_squared = np.mean(squared)
    root_mean_squared = np.sqrt(mean_squared)
    #print('rms',root_mean_squared)
    try:
        dB_rms = 20.0*np.log10(root_mean_squared)
    except ValueError:
        dB_rms = -np.inf
    return(dB_rms)

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put((0.053+indata).copy())
    args.buffer = ((0.053+indata).copy())

    #print('buffer=',np.shape(args.buffer))

current_frame = 0
def callback_out(outdata, frames, time, status):
    global current_frame
    if status:
        print(status)
    chunksize = min(len(data) - current_frame, frames)
    outdata[:chunksize] = data[current_frame:current_frame + chunksize]
    #audio_in_data.put(indata.copy())
    if chunksize < frames:
        outdata[chunksize:] = 0
        raise sd.CallbackStop()
    current_frame += chunksize
    
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

def calc_delay(two_ch,fs):
    
    '''
    Parameters
    ----------
    two_ch : (Nsamples, 2) np.array
        Input audio buffer
    ba_filt : (2,) tuple
        The coefficients of the low/high/band-pass filter
    fs : int, optional
        Frequency of sampling in Hz. Defaults to 44.1 kHz

    Returns
    -------
    delay : float
        The time-delay in seconds between the arriving audio across the 
        channels. 
    '''
    for each_column in range(2):
        two_ch[:,each_column] = two_ch[:,each_column]

    cc = np.correlate(two_ch[:,0],two_ch[:,1],'same')
    midpoint = cc.size/2.0
    delay = np.argmax(cc) - midpoint
    # convert delay to seconds
    delay *= 1/float(fs)
    return delay

def gcc_phat(sig,refsig, fs):
    # Compute the cross-correlation between the two signals
    #sig = sig[:,1]
    
    n = sig.shape[0] + refsig.shape[0]
    SIG = fft(sig, n=n)
    REFSIG = fft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.ifft(R / np.abs(R))
    max_shift = int(np.floor(n / 2))
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    #plt.plot(cc)
    #plt.show()
    #plt.title('gcc-phat')
    shift = np.argmax(np.abs(cc)) - max_shift
    return -shift / float(fs)

def calc_multich_delays(multich_audio,ref_sig,fs):
    '''s
    Calculates peak delay based with reference of 
    channel 1. 
    '''
    nchannels = multich_audio.shape[1]
    delay_set = []
    delay_set_gcc = []
    i=0
    while i < nchannels:
        if i != ref:
            #print(i)
            
            #delay_set.append(calc_delay(multich_audio[:,[ref, i]],fs)) #cc without phat norm
            delay_set.append(gcc_phat(multich_audio[:,i],ref_sig,fs)) #gcc phat correlation
            i+=1
        else:
            #print('else',i)
            i+=1
            pass

    #print('delay=',delay_set)
    #print('delay gcc=',delay_set_gcc)
    return np.array(delay_set)

def avar_angle(delay_set,nchannels,mic_spacing):
    '''
    calculates the mean angle of arrival to the array
    with channel 1 as reference
    '''
    theta = []
    #print('delay_set=', delay_set)
    if ref!=0: #centered reference that works with odd mics
        for each in range(0, nchannels//2):
            #print('\n1',each)
            #print('1',nchannels//2-each)
            theta.append(-np.arcsin((delay_set[each]*343)/((nchannels//2-each)*mic_spacing))) # rad
            i=nchannels//2-each
            #print('i=',i)
        for each in range(nchannels//2, nchannels-1):
            #print('\n2',each)
            #print('2',i)
            theta.append(np.arcsin((delay_set[each]*343)/((i)*mic_spacing))) # rad
            i+=1
    else:   
        for each in range(0, nchannels-1):
            theta.append(np.arcsin((delay_set[each]*343)/((each+1)*mic_spacing))) # rad
    #print('theta',theta)
    avar_theta = np.mean(theta)
    return avar_theta
   
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def save_data_to_csv(matrix, filename, path):
    full_path = os.path.join(path, filename)
    with open(full_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(matrix)
    print(f"Matrix has been saved as csv to {full_path}")

def save_data_to_xml(matrix, filename, path):
    full_path = os.path.join(path, filename)
    root = ET.Element("matrix")
    for row in matrix:
        row_elem = ET.SubElement(root, "row")
        for val in row:
            val_elem = ET.SubElement(row_elem, "val")
            val_elem.text = str(val)
    
    tree = ET.ElementTree(root)
    tree.write(full_path)
    print(f"Matrix has been saved as xml to {full_path}")

avar_theta = None
theta_values   = []

def update():

    in_sig = args.buffer

    ref_channels = in_sig
    #print(np.shape(in_sig))
    #print(in_sig)
    #print('ref_channels=', np.shape(ref_channels))
    ref_channels_bp = bandpass_sound(ref_channels,a,b)
    #print('ref_channels_bp=', np.shape(ref_channels_bp))
    above_level,dBrms_channel = check_if_above_level(ref_channels_bp,trigger_level)
    #print(above_level)
    av_above_level = np.mean(dBrms_channel)
    print(av_above_level)
    ref_sig = in_sig[:,ref]
    delay_crossch= calc_multich_delays(in_sig,ref_sig, fs)

    # calculate avarage angle
    avar_theta = avar_angle(delay_crossch,channels,mic_spacing)
    time3 = datetime.datetime.now()
    avar_theta1 = np.array([avar_theta, time3.strftime('%H:%M:%S.%f')[:-3]])

    #print('avarage theta',avar_theta1)

    qqq.put(avar_theta1.copy()) # store detected angle value
    #if av_above_level > critical_level:
    #    avar_theta = critical
    #    return avar_theta
        
    if av_above_level > trigger_level:


        #calculate avarage angle
        #avar_theta = avar_angle(delay_crossch,channels,mic_spacing)
        #avar_theta = avar_angle(delay_crossch_gcc,channels,mic_spacing)
        
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
    #print('azimuth_recon=',doa.azimuth_recon) # rad value of detected angles
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
    ref_channels_bp = bandpass_sound(ref_channels,a,b)
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

def main(use_sim=False, ip='localhost', port=2001):
    ''' Main function '''
    try:
        global startime
        # Configure Interface to Thymio robot
        # simulation
        if use_sim:
            th = Thymio(use_tcp=True, host=ip, tcp_port=port, 
                        on_connect=lambda node_id: print(f' Thymio {node_id} is connected'))
        # real robot
        else:
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

        # Make sure the file is opened before recording anything:
        with sd.InputStream(samplerate=args.samplerate, device=usb_fireface_index,
                            channels=args.channels, callback=callback, blocksize=block_size) as input_stream:
            print('audio stream started')
            #time.sleep(1)
            #print('sleep')
            waiturn = 0.1
            wait = 0
            start_time_rec = time.time()
            #print(start_time_rec)
            start_time = time.time()
            rec_counter = 0
            pause = False

            while True:
                if (time.time() - start_time_rec) <= 5: #seconds
                    pass
                else:
                    #input_stream.stop()
                    #print('stop')
                    #break

                    #record(rec_counter,time1)
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
                    rec_counter += 1
                    start_time_rec = time.time()
                    record(rec_counter,time1)
                    # print('delta save=',time.time()-start_time_rec,'sec')
                    #print('startime = ',start_time_rec)

                # Check if the elapsed time since the last frame is less than or equal to the desired frame duration
                #if (time.time() - start_time) <= 1/fps: 
                #    pass
                #else:
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
                    else:
                        #print("Warning: angle is not a numerical type")
                        pass
                    print('Angle = ', angle)
                    print('av ab level = ', av_above_level)
                    #angle = int(angle)
                    
                    #print('updatestart=',datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S'))
                    time_end = time.time()
                    #print('delta update cc',time_end - time_start,'sec')

                else:
                    print('No valid method provided')

                ground_sensors = robot['prox.ground.reflected']
                #print('ground = ',robot['prox.ground.reflected'])
                
		# Adjust these threshold values as needed
                left_sensor_threshold = 300
                right_sensor_threshold = 300		
                
                norm_coefficient = 48000/1024 #most used fractional value, i.e. normalization value 
                max_speed = 400 #to be verified 
                speed = 0.5 * max_speed * analyzed_buffer *(1/norm_coefficient) #generic speed of robot while moving 
                speed = int(speed)
                #speed = 100
                print('\nspeed = ',speed, '\n')
                if angle != None:
                    turn_speed = 1/90 * (speed*int(angle)) #velocity of the wheels while turning 
                    turn_speed = int(turn_speed)

                    print('\nturn_speed = ',turn_speed, '\n')
                #turn_speed = 100
                direction = random.choice(['left', 'right'])
                
                #PROPORTIONAL MOVEMENT
                if ground_sensors[0] > left_sensor_threshold  and ground_sensors[1]> right_sensor_threshold:
                    # Both sensors detect the line, turn left
                    if direction == 'left':
                        robot['motor.left.target'] = -speed
                        robot['motor.right.target'] = speed   
                        time.sleep(0.5) 
                        pass
                    else:
                        robot['motor.left.target'] = speed
                        robot['motor.right.target'] = -speed
                        time.sleep(0.5)
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
                        robot['motor.left.target'] =   - turn_speed
                        robot['motor.right.target'] =  + turn_speed
                        time.sleep(waiturn)
                    else: #attraction
                        if angle < 0:
                            robot['motor.left.target'] =   speed + turn_speed
                            robot['motor.right.target'] =  speed - turn_speed
                            time.sleep(waiturn)
                        else:
                            robot['motor.left.target'] = speed + turn_speed
                            robot['motor.right.target'] = speed - turn_speed
                            time.sleep(waiturn)
                
                #NON PROPORTIONAL MOVEMENT

#                if angle == critical:
#                    # critical distance: take right or left avoidance
#
#                    if direction == 'left':
#                        robot['motor.left.target'] = -200
#                        robot['motor.right.target'] = 200   
#                        time.sleep(0.5) 
#                        pass
#                    else:
#                        robot['motor.left.target'] = 200
#                        robot['motor.right.target'] = -200
#                        time.sleep(0.5)
#                        pass
#                elif ground_sensors[0] > left_sensor_threshold  and ground_sensors[1]> right_sensor_threshold:
#                    # Both sensors detect the line, turn left
#                    if direction == 'left':
#                        robot['motor.left.target'] = -280
#                        robot['motor.right.target'] = 230   
#                        time.sleep(0.5) 
#                        pass
#                    else:
#                        robot['motor.left.target'] = 230
#                        robot['motor.right.target'] = -280
#                        time.sleep(0.5)
#                        pass
#                elif ground_sensors[1] > right_sensor_threshold:
#                    # Only right sensor detects the line, turn left
#                    robot['motor.left.target'] = -150
#                    robot['motor.right.target'] = 150
#                    time.sleep(waiturn)
#                elif ground_sensors[0] > left_sensor_threshold:
#                    # Only left sensor detects the line, turn right
#                    robot['motor.left.target'] = 150 
#                    robot['motor.right.target'] = -150 
#                    time.sleep(waiturn)
#                else:       
#                    match angle:
#                        case theta if theta == None:
#                            robot["leds.top"] = [0, 0, 0]
#                            time.sleep(wait)
#                            robot['motor.left.target'] = 200
#                            robot['motor.right.target'] = 200
#                        case theta if theta < -30:
#                            robot["leds.top"] = [0, 0, 255]
#                            time.sleep(wait)
#                            robot['motor.left.target'] = 300
#                            robot['motor.right.target'] = 20
#                            time.sleep(wait)
#                        case theta if -30 <= theta < -5:
#                            robot["leds.top"] = [0, 255, 255]
#                            time.sleep(wait)
#                            robot['motor.left.target'] = 200
#                            robot['motor.right.target'] = 20
#                            time.sleep(wait)
#                        case theta if -5 <= theta < -1:
#                            robot["leds.top"] = [0, 255, 0]
#                            time.sleep(wait)
#                            robot['motor.left.target'] = 50
#                            robot['motor.right.target'] = 10
#                            time.sleep(wait)
#                        case theta if 0 <= theta <= 0:
#                            robot["leds.top"] = [255, 255, 255]
#                            time.sleep(wait)
#                            robot['motor.left.target'] = 40
#                            robot['motor.right.target'] = 40
#                            time.sleep(0.1)
#                            #direction = random.choice(['left', 'right'])
#                            #if direction == 'left':
#                            #    robot['motor.left.target'] = -150
#                            #    robot['motor.right.target'] = 150
#                            #    time.sleep(0.1)
#                            #    pass
#                            #else:
#                            #    robot['motor.left.target'] = 150
#                            #    robot['motor.right.target'] = -150
#                            #    time.sleep(0.1)
#                            #    pass
#                            #time.sleep(waiturn)
#                        case theta if 1 < theta <= 5:
#                            robot["leds.top"] = [0, 255, 0]
#                            time.sleep(wait)
#                            robot['motor.right.target'] = 50
#                            robot['motor.left.target'] = 10
#                            time.sleep(wait)
#                        case theta if 5 < theta <= 30:
#                            robot["leds.top"] = [255, 255, 0]
#                            time.sleep(wait)
#                            robot['motor.right.target'] = 200
#                            robot['motor.left.target'] = 20
#                            time.sleep(wait)
#                        case theta if theta > 30:
#                            robot["leds.top"] = [255, 0, 0]
#                            time.sleep(wait)
#                            robot['motor.right.target'] = 300
#                            robot['motor.left.target'] = 20
#                            time.sleep(wait)
#                        case _:
#                            pass
#

                # print('delta loop =',time.time()-start_time,'sec')
                start_time = time.time()

        with sd.OutputStream(samplerate=fss,
                            blocksize=0, 
                            device=usb_fireface_index, 
                            channels=1,
                            callback=callback_out,
                            latency='low') as stream:
                            while stream.active:
                                pass

    except Exception as err:
        # Stop robot
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0 
        robot["leds.top"] = [0,0,0]
        print(err)
    except KeyboardInterrupt:
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0
        robot["leds.top"] = [0,0,0]
        print("Press Ctrl-C again to end the program")    

        save_path = '/home/thymio/robat_py/robat_v0_files/'
        #save_path = ''
        folder_name = str(time1) + '_rec_data' 
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        if method == 'CC':
            theta_doa = np.vstack([[qqq.get() for _ in range(qqq.qsize())]]) #recognized angle values
            file_theta_doa_xml = time1 + "_theta_doa_CC.xml"
            file_theta_doa_csv = time1 + "_theta_doa_CC.csv"
            print(np.shape(theta_doa))
            print(theta_doa)
            #save_data_to_xml(theta_doa, file_theta_doa_xml, folder_path) #[time, av angle deg]
            #save_data_to_csv(theta_doa, file_theta_doa_csv, folder_path) #[time, av angle deg]
            
        if method == 'PRA':
            spatial_response = [qq.get() for _ in range(qq.qsize())] #360 values for plot
            print(np.shape(spatial_response))

            theta_doa = [[qqq.get() for _ in range(qqq.qsize())]] #recognized angle values

            file_spat_resp_xml = time1 + "_spat_resp_PRA.xml" 
            file_spat_resp_csv =  time1 + "_spat_resp_PRA.csv" 
            #save_data_to_xml(spatial_response, file_spat_resp_xml, folder_path) #[time, 360 polar plot angles]
            #save_data_to_csv(spatial_response, file_spat_resp_csv, folder_path) #[time, 360 polar plot angles]
            
            file_theta_doa_xml = time1 + "_theta_doa_PRA.xml" #[time, number of detected angle]
            file_theta_doa_csv = time1 + "_theta_doa_PRA.csv" #[time, number of detected angle]
            #save_data_to_xml(theta_doa, file_theta_doa_xml, folder_path)
            #save_data_to_csv(theta_doa, file_theta_doa_csv, folder_path)

        
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
#        sd.stop()
        

print('functions loaded')

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

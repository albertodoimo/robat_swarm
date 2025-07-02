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

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)

block_size = 1024
q = queue.Queue()

rand = random.uniform(0.8, 1.2)
duration_out = 2  # Duration in seconds
duration_in = rand * 0.5  # Duration in seconds
amplitude = 0.1
trigger_level = -65 # dB level ref max pdm
critical_level = -50 # dB level pdm critical distance
critical = []
mic_spacing = 0.018 #m

c = 343   # speed of sound
fs = 48000
rec_samplerate = 44000
block_size = 1024
analyzed_buffer = fs/block_size #theoretical buffers analyzed each second 
channels = 5
mic_spacing = 0.018 #m
ref = channels//2 #central mic in odd array as ref
print('ref=',ref) 

method = 'CC'
audio_file = '/home/thymio/robat_py/untitled.mp3'
#data, fs = sf.read(audio_file)
#data =  amplitude * data[:fs*duration_out,1].reshape(-1,1)

t = np.arange(fs * duration_out) / fs  # Time vector 
window = np.hanning(len(t))
#data = (amplitude * np.sin(2 * np.pi * 5000 * t) * window).reshape(-1, 1)
#t = np.linspace(0, duration_out, int(fs * duration_out))
data = signal.chirp(t, f0=20000, f1=200, t1=duration_out, method='linear')

data = amplitude * np.random.normal(0, 1, (fs * duration_out, 1)) 

auto_hipas_freq = int(343/(2*(mic_spacing*(channels-1))))
print('HP', auto_hipas_freq)
highpass_freq, lowpass_freq = [20 ,21000]
freq_range = [highpass_freq, lowpass_freq]

nyq_freq = fs/2.0
b, a = signal.butter(4, [highpass_freq/nyq_freq,lowpass_freq/nyq_freq],btype='bandpass') # to be 'allowed' in Hz.

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
    if status:
        print(status, file=sys.stderr)
    q.put((0.053+indata).copy())
    args.buffer = ((0.053+indata).copy())

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
    #time3 = datetime.datetime.now()
    #avar_theta1 = np.array([avar_theta, time3.strftime('%H:%M:%S.%f')[:-3]])

    #print('avarage theta',avar_theta1)

    #qqq.put(avar_theta1.copy()) # store detected angle value
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

def main(sim, ip, port):
    try:
        if sim:
            # Simulation code here
            pass
        else:
            port = Connection.serial_default_port()
            th = Thymio(serial_port=port, 
                        on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
            # Connect to Robot
        # Connect to Robot
        th.connect()
        robot = th[th.first_node()]

        args.samplerate = fs
        if args.samplerate is None:  
            print('error!: no samplerate set! Using default')
            device_info = sd.query_devices(args.device, 'input')
            args.samplerate = int(device_info['default_samplerate'])
        #if args.filename is None:
        #    timenow = datetime.datetime.now()
        #    time1 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
        #    args.filename = 'MULTIWAV_' + str(time1) + '.wav'
        #print(args.samplerate)
        while True: 
            current_frame = 0       
            with sd.OutputStream(samplerate=fs,
                                blocksize=0, 
                                device=usb_fireface_index, 
                                channels=1,
                                callback=callback_out,
                                latency='low') as out_stream:
                                while out_stream.active:
                                    pass
            #out_stream.stop()
            start_time = time.time()

            with sd.InputStream(samplerate=fs, device=usb_fireface_index,channels=channels, callback=callback_in, blocksize=block_size) as input_stream:
                while time.time() - start_time < duration_in:
                    print('time =', time.time() - start_time)

                    if method == 'CC':
                        
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
                    print('ground = ',robot['prox.ground.reflected'])
                    
                    # Adjust these threshold values as needed
                    left_sensor_threshold = 600
                    right_sensor_threshold = 600		
                    
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

                    elif ground_sensors[0] > left_sensor_threshold:
                        # Only left sensor detects the line, turn right
                        robot['motor.left.target'] = speed 
                        robot['motor.right.target'] = -speed 

                    else:  #attraction or repulsion 
                        if angle == None: #neutral movement
                            robot["leds.top"] = [0, 0, 0]
                            robot['motor.left.target'] = speed
                            robot['motor.right.target'] = speed
                            
                        elif av_above_level > critical_level: #repulsion
                            robot['motor.left.target'] =   - turn_speed
                            robot['motor.right.target'] =  + turn_speed

                        else: #attraction
                            if angle < 0:
                                robot['motor.left.target'] =   speed + turn_speed
                                robot['motor.right.target'] =  speed - turn_speed

                            else:
                                robot['motor.left.target'] = speed + turn_speed
                                robot['motor.right.target'] = speed - turn_speed

                else:
                    robot['motor.left.target'] = 0
                    robot['motor.right.target'] = 0
                    #input_stream.stop()

    except Exception as err:
        print(f'Error: {err}')
        sys.exit(1)
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

if __name__ == '__main__':
    # Parse commandline arguments to configure the interface for a simulation (default = real robot)
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                 'If no arguments are given, the code will run with a real Thymio.', add_help=False)
    
    # Add optional arguments
    parser.add_argument('-s', '--sim', action='store_true', help='set this flag to use simulation')
    parser.add_argument('-i', '--ip', help='set the TCP host ip for simulation. default=localhost', default='localhost')
    parser.add_argument('-p', '--port', type=int, help='set the TCP port for simulation. default=2001', default=2001)
    parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio devices and exit')
    
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[parser])
    args = parser.parse_args(remaining)

    # Initialize buffer
    args.buffer = np.zeros((block_size, channels))
    
    # Call main function with parsed arguments
    main(args.sim, args.ip, args.port)


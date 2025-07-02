
print('libraries installed')
print('import libraries...')

import argparse
import time
import math
import random

# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# from pyqtgraph.Qt import QtCore
import sounddevice as sd
import numpy as np
from scipy import signal
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# thymio
from thymiodirect import Connection 
from thymiodirect import Thymio

print('libraries imported')

print('loading functions...')
#%%
def calc_rms(in_sig):
    '''
    
    '''
    rms_sig = np.sqrt(np.mean(in_sig**2))
    # print('rms =', rms_sig)
    return(rms_sig)

def calc_rms_avar(in_sig,ch):
    rms_sig = []
    #print(ch)
    #print('empty rms =', rms_sig)
    for i in range(ch):
        #print(i)
        rms_sig.append(np.sqrt(np.mean(in_sig[:,i]**2)))
        #print('rms =', rms_sig)
    
    avar_rms = np.mean(rms_sig)
    #print('rms avar =', avar_rms)
    return(avar_rms)
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
    # if np.abs(delay)< 5.5*10**-5:
    #     delay = 0
    # else:
    #     delay = delay
    return delay

def calc_multich_delays(multich_audio,fs):
    '''s
    Calculates peak delay based with reference of 
    channel 1. 
    '''
    nchannels = multich_audio.shape[1]
    delay_set = []
    for each in range(1, nchannels):
        delay_set.append(calc_delay(multich_audio[:,[0,each]],fs))
    # print(delay_set)
    return np.array(delay_set)

def avar_angle(delay_set,nchannels,mic_spacing):
    '''
    calculates the mean angle of arrival to the array
    with channel 1 as reference
    '''
    theta = []
    for each in range(0, nchannels-1):
        theta.append(np.arcsin((delay_set[each]*343)/((each+1)*mic_spacing))) # rad
    # print('theta=',theta)
    avar_theta = np.mean(theta)
    return avar_theta

print('functions loaded')

print('initializating audio stream...')
#%% Set up the audio-stream of the laptop, along with how the 
# incoming audio buffers will be processed and thresholded.
import queue
input_audio_queue = queue.Queue()

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)
fs = 96000
block_size = 4096
# block_size = 1024*2
#block_size = 8192
nchannels = 5
mic_spacing = 0.018 #m
central_mic = 3

#bp_freq = np.array([100,45000.0]) # the min and max frequencies
# to be 'allowed' in Hz.

# ba_filt = signal.butter(2, bp_freq/float(fs*0.5),'bandpass')

#%%
# define the input signals features
S = sd.InputStream(samplerate=fs,blocksize=block_size, device=usb_fireface_index, channels=nchannels, latency='low')
print('fs = ', S.samplerate)
print('blocksize = ', S.blocksize)
print('channels = ', S.channels)
print('latency = ', S.latency)
print('devinfo = ', S.device)
S.start()

# creation the guide vector x values
# all_xs = np.linspace(-10,10,S.blocksize)
# print('all_xs',all_xs.shape)
# threshold = 1e-5

print('audio stream initialized')

in_sig,status = S.read(S.blocksize)
#threshold = calc_rms_avar(in_sig, nchannels)
threshold = calc_rms_avar(in_sig, nchannels)* 1.005

print('thresh=', threshold)

def update():
    #global sp_my, all_xs, threshold, S, ba_filt
    try:
        in_sig,status = S.read(S.blocksize)
        #rms_sig = calc_rms(in_sig[:,central_mic])
        rms_sig = calc_rms_avar(in_sig, nchannels)
        print('rms_central mic = ', rms_sig)
        if rms_sig > threshold:

            delay_crossch = calc_multich_delays(in_sig, fs)
    

            # print('delay',delay_crossch)
            # calculate aavarage angle

            avar_theta = avar_angle(delay_crossch,nchannels,mic_spacing)
            return np.rad2deg(avar_theta)
        else:
            avar_theta = None
            return avar_theta
    except KeyboardInterrupt:
        S.stop()



# Thymio 
# # %%------------------------------------------------------
# 
wait = 0.0005
waiturn = 0.01
def main(use_sim=False, ip='localhost', port=2001):
    ''' Main function '''

    try:
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

        # Delay to allow robot initialization of all variables
        #time.sleep(1)
        while True:
            avar_theta_deg = update()
            # avar_theta_deg = avar_theta_deg*1.25
            #detectsCollision = max([robot['prox.horizontal'][i] > 1200 for i in range(5)])
            print('avarage theta deg', avar_theta_deg)

            ground_sensors = robot['prox.ground.reflected']
            ground_sensors_max = 1000
            # Adjust these threshold values as needed
            ground_sensors = robot['prox.ground.reflected']
            #print('ground = ',robot['prox.ground.reflected'])
            # Adjust these threshold values as needed
            left_sensor_threshold = 80
            right_sensor_threshold = 80
            direction = random.choice(['left', 'right'])
            if ground_sensors[0] > left_sensor_threshold  and ground_sensors[1]> right_sensor_threshold:
                # Both sensors detect the line, turn left
                if direction == 'left':
                    robot['motor.left.target'] = -150
                    robot['motor.right.target'] = 150   
                    time.sleep(waiturn) 
                else:
                    robot['motor.left.target'] = 150
                    robot['motor.right.target'] = -150
                    time.sleep(waiturn)
                # robot['motor.left.target'] = -50 + random.choice([, 100])
                # robot['motor.right.target'] = -50 + random.choice([-100, 100])
            elif ground_sensors[1] > right_sensor_threshold:
                # Only right sensor detects the line, turn left
                robot['motor.left.target'] = -150
                robot['motor.right.target'] = 150
                time.sleep(waiturn)
            elif ground_sensors[0] > left_sensor_threshold:
                # Only left sensor detects the line, turn right
                robot['motor.left.target'] = 150 
                robot['motor.right.target'] = -150 
                time.sleep(waiturn)
            else:       
                match avar_theta_deg:
                    case theta if theta == None:
                        robot["leds.top"] = [0, 0, 0]
                        time.sleep(wait)
                        robot['motor.left.target'] = 200
                        robot['motor.right.target'] = 200
                    case theta if theta < -30:
                        robot["leds.top"] = [0, 0, 255]
                        time.sleep(wait)
                        robot['motor.left.target'] = 400
                        robot['motor.right.target'] = 20
                        time.sleep(wait)
                    case theta if -30 <= theta < -5:
                        robot["leds.top"] = [0, 255, 255]
                        time.sleep(wait)
                        robot['motor.left.target'] = 300
                        robot['motor.right.target'] = 20
                        time.sleep(wait)
                    case theta if -5 <= theta <= 5:
                        robot["leds.top"] = [255, 255, 255]
                        time.sleep(wait)
                        robot['motor.left.target'] = -100
                        robot['motor.right.target'] = -100
                        time.sleep(waiturn)
                        direction = random.choice(['left', 'right'])
                        if direction == 'left':
                            robot['motor.left.target'] = -150
                            robot['motor.right.target'] = 150
                        else:
                            robot['motor.left.target'] = 150
                            robot['motor.right.target'] = -150
                        time.sleep(waiturn)
                    case theta if 5 < theta <= 30:
                        robot["leds.top"] = [255, 255, 0]
                        time.sleep(wait)
                        robot['motor.right.target'] = 300
                        robot['motor.left.target'] = 20
                        time.sleep(wait)
                    case theta if theta > 30:
                        robot["leds.top"] = [255, 0, 0]
                        time.sleep(wait)
                        robot['motor.right.target'] = 400
                        robot['motor.left.target'] = 20
                        time.sleep(wait)
                    case _:
                        pass
                    # if ground_sensors[0] > left_sensor_threshold  and ground_sensors[1]> right_sensor_threshold:
                    #     # Both sensors detect the line, turn left
                    #     robot['motor.left.target'] = -100
                    #     robot['motor.right.target'] = 100
                    # elif ground_sensors[0] < left_sensor_threshold and ground_sensors[1] > right_sensor_threshold:
                    #     # Only right sensor detects the line, turn left
                    #     robot['motor.left.target'] = -100
                    #     robot['motor.right.target'] = 100
                    # elif ground_sensors[0] > left_sensor_threshold and ground_sensors[1] < right_sensor_threshold:
                    #     # Only left sensor detects the line, turn right
                    #     robot['motor.left.target'] = 100 
                    #     robot['motor.right.target'] = -100  
    
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
    # time.sleep(0.0005)
if __name__ == '__main__':
    # Parse commandline arguments to cofigure the interface for a simulation (default = real robot)
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                    'If no arguments are given, the code will run with a real Thymio.')
    
    # Add optional arguments
    parser.add_argument('-s', '--sim', action='store_true', help='set this flag to use simulation')
    parser.add_argument('-i', '--ip', help='set the TCP host ip for simulation. default=localhost', default='localhost')
    parser.add_argument('-p', '--port', type=int, help='set the TCP port for simulation. default=2001', default=2001)

    # Parse arguments and pass them to main function
    args = parser.parse_args()
    main(args.sim, args.ip, args.port)

print('import libraries...')

import argparse
import time
import math
import random
import datetime
import os

# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# from pyqtgraph.Qt import QtCore
import sounddevice as sd
import soundfile as sf
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
# def calc_rms(in_sig):
#     '''
#     
#     '''
#     rms_sig = np.sqrt(np.mean(in_sig**2))
#     return(rms_sig)
# 

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
fs = 48000
block_size = 4096
channels = 8
mic_spacing = 0.003 #m

#%%
# define the input signals features
S = sd.InputStream(samplerate=fs,blocksize=block_size,channels=channels, latency='low', device=usb_fireface_index)
print('fs = ', S.samplerate)
print('blocksize = ', S.blocksize)
print('channels = ', S.channels)
print('latency = ', S.latency)
print('devinfo = ', S.device)
S.start()

memory = []
rec = []

print('audio stream initialized')

def update():
    #global sp_my, all_xs, threshold, S, ba_filt
    try:
        global memory

        in_sig,status = S.read(S.blocksize)

        memory.append(in_sig)

        print('input audio plot = ', np.shape(memory))
        
        # Filter input signal
        # delay_crossch = calc_multich_delays(in_sig,ba_filt,fs)
        # delay_crossch = calc_multich_delays(in_sig[:,[2,3,4,5]],ba_filt,fs)
        delay_crossch = calc_multich_delays(in_sig[:,[0,1,2,3,4,5,6,7]],fs)

        # print('delay',delay_crossch)
        # calculate aavarage angle

        avar_theta = avar_angle(delay_crossch,channels-1,mic_spacing)
        #print('avarage theta rad',avar_theta)
        # print('avarage theta deg',np.rad2deg(avar_theta))
        
    except KeyboardInterrupt:

        print("update function stopped")

        S.stop()
        # print('memory shape =', np.shape(memory))
        rec = np.concatenate(memory)
        # print('memory shape =', np.shape(rec))

        stoptime = datetime.datetime.now()
        print("\nREC START TIME: \n", startime.strftime("%Y-%m-%d %H:%M:%S"))
        print("\nSTOP REC TIME: \n", stoptime.strftime("%Y-%m-%d %H:%M:%S"))
        print('')

        if not os.path.exists('recordings'):
            os.makedirs('recordings')

        os.makedirs(f'recordings/rec_{startime}')
        #time.sleep(1000)
        for i in range(channels):
            
            sf.write(f'recordings/rec_{startime}/{startime}_AudioRec_Ch_{i+1}.wav', rec[:,i], samplerate=fs)
        
        


    return np.rad2deg(avar_theta)

# Thymio 
# # %%------------------------------------------------------
# 
wait = 0.0001
waiturn = 0.3
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

        # Delay to allow robot initialization of all variables
        #time.sleep(1)


        startime = datetime.datetime.now()
        print("\nREC START TIME: \n", startime.strftime("%Y-%m-%d %H:%M:%S"))
        print('')
        
        while True:
            avar_theta_deg = update()
            avar_theta_deg = avar_theta_deg*1.25
            #detectsCollision = max([robot['prox.horizontal'][i] > 1200 for i in range(5)])
            print('avarage theta deg', avar_theta_deg)

            ground_sensors = robot['prox.ground.reflected']
            ground_sensors_max = 1000
            # Adjust these threshold values as needed
            ground_sensors = robot['prox.ground.reflected']
            print('ground = ',robot['prox.ground.reflected'])
            # Adjust these threshold values as needed
            left_sensor_threshold = 800
            right_sensor_threshold = 800
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
                    case theta if theta < -30:
                        robot["leds.top"] = [0, 0, 255]
                        time.sleep(wait)
                        robot['motor.left.target'] = 400
                        robot['motor.right.target'] = 20
                        time.sleep(wait)
                    case theta if -30 <= theta < -1:
                        robot["leds.top"] = [0, 255, 255]
                        time.sleep(wait)
                        robot['motor.left.target'] = 300
                        robot['motor.right.target'] = 20
                        time.sleep(wait)
                    case theta if -1 <= theta <= 1:
                        robot["leds.top"] = [255, 255, 255]
                        time.sleep(wait)
                        robot['motor.left.target'] = 200
                        robot['motor.right.target'] = 200
                        time.sleep(wait)
                    case theta if 1 < theta <= 30:
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
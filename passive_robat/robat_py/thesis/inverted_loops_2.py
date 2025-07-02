#!/usr/bin/env python3
"""Create a recording with arbitrary duration.

The soundfile module (https://python-soundfile.readthedocs.io/)
has to be installed!

"""
print('import libraries...')
import argparse
import tempfile
import queue
import sys
import json
import datetime
import time
import argparse
import math
import random
import os
from thymiodirect import Connection 
from thymiodirect import Thymio
from scipy import signal


import sounddevice as sd
import soundfile as sf
import numpy  as np 

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

fs = 48000
rec_samplerate = 48000
block_size = 1024
channels = 7
mic_spacing = 0.015 #m


auto_hipas_freq = int(343/(2*(mic_spacing*(channels-1))))
auto_lowpas_freq = int(343/(2*mic_spacing))
#print(auto_hipas_freq)
highpass_freq, lowpass_freq = [auto_hipas_freq ,auto_lowpas_freq]
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
trigger_level = -50 # dB level ref max pdm


print('loading functions...')

q = queue.Queue()

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
    try:
        dB_rms = 20.0*np.log10(root_mean_squared)
    except:
        dB_rms = -999.
    return(dB_rms)

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())
    
    args.buffer = (indata.copy())

    #print('buffer=',np.shape(args.buffer))

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
   
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

avar_theta = None
def update():
    #try:
        in_sig = args.buffer
#        #print(in_sig)
#        # calculate avarage angle
#        delay_crossch = calc_multich_delays(in_sig,fs)
#        avar_theta = avar_angle(delay_crossch,channels-1,mic_spacing)
#        #print('avarage theta rad',avar_theta)
#        # print('avarage theta deg',np.rad2deg(avar_theta))

        ref_channels = in_sig
        #print('ref_channels=', np.shape(ref_channels))
        ref_channels_bp = bandpass_sound(ref_channels,a,b)
        #print('ref_channels_bp=', np.shape(ref_channels_bp))
        above_level,dBrms_channel = check_if_above_level(ref_channels_bp,trigger_level)
        #print(above_level)
        av_above_level = np.mean(dBrms_channel)
        #print(av_above_level)
        if av_above_level>trigger_level:
            #print(av_above_level)

            delay_crossch = calc_multich_delays(ref_channels_bp, fs)
#     
# 
#             # print('delay',delay_crossch)
#             # calculate aavarage angle
# 
            avar_thetaa = avar_angle(delay_crossch,channels,mic_spacing)
            print(np.rad2deg(avar_thetaa))
            #return np.rad2deg(avar_theta)
        else:
            avar_theta = None
            return avar_theta
        
#    except KeyboardInterrupt:
#        
#
#        print("\nupdate function stopped\n")
#        
#        q_contents = [q.get() for _ in range(q.qsize())]
#        print('q_contents = ', np.shape(q_contents))
#        rec = np.concatenate(q_contents)
#        print('rec = ', np.shape(rec))
#        rec2besaved = rec[:, :channels]
#        save_path = '/home/thymio/robat_py/recordings/'
#        full_path = os.path.join(save_path, args.filename)
#        with sf.SoundFile(full_path, mode='x', samplerate=args.samplerate,
#                        channels=args.channels, subtype=args.subtype) as file:
#            file.write(rec2besaved)
#            print(f'\nsaved to {args.filename}\n')
#        sd.stop() 
#
        #return np.rad2deg(avar_theta)


def main(use_sim=False, ip='localhost', port=2001):
    ''' Main function '''
    try:
        global startime, process
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
    
        #print("\nREC START TIME: \n", startime.strftime("%Y-%m-%d %H:%M:%S"))
        #print('')

        if args.samplerate is None:    
            device_info = sd.query_devices(args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info['default_samplerate'])
        if args.filename is None:
            #args.filename = tempfile.mktemp(prefix='delme_rec_unlimited_',suffix='.wav', dir='')
            timenow = datetime.datetime.now()
            time1 = timenow.strftime('%d-%m-%Y_%H-%M-%S')
            args.filename = 'MULTIWAV_inverted_loops_' + str(time1) + '.wav'

        # Make sure the file is opened before recording anything:
        with sd.InputStream(samplerate=args.samplerate, device=usb_fireface_index,
                            channels=args.channels, callback=callback, blocksize=block_size):
            #print('#' * 80)
            #print('press Ctrl+C to stop the recording')
            #print('#' * 80)
            print('audio stream started')
            waiturn = 0.8
            wait = 0.0005
            start_time = time.time()
            rec_counter = 1
            while True:

                
            #    # This loop runs for 1 second after the stream is started
            #    #print(time.time() - start_time)
            #    #print(start_time)©
            #    #print(time.time())
            #    if (time.time() - start_time) <=30: #seconds
            #        pass
            #    else:
            #        q_contents = [q.get() for _ in range(q.qsize())]
            #        
            #        #time.sleep(1)
            #        print('q_contents = ', np.shape(q_contents))
            #        rec = np.concatenate(q_contents)
            #        print('rec = ', np.shape(rec))
            #        
#
            #        rec2besaved = rec[:, :channels]
            #        save_path = '/home/thymio/robat_py/recordings/' 
            #        # Create folder with args.filename (without extension)
            #        folder_name= 'MULTIWAV_inverted_loops_' + str(time1) 
            #        folder_path = os.path.join(save_path, folder_name)
            #        os.makedirs(folder_path, exist_ok=True)
            #        save_path = folder_path
#
            #        timenow = datetime.datetime.now()
            #        time2 = timenow.strftime('%d-%m-%Y_%H-%M-%S')
            #        full_path = os.path.join(save_path, str(rec_counter)+'.wav')
            #        with sf.SoundFile(full_path, mode='x', samplerate=rec_samplerate,
            #                        channels=args.channels, subtype=args.subtype) as file:
            #            file.write(rec2besaved)
            #            print(f'\nsaved to {full_path}\n')
            #            rec_counter += 1
            #        start_time = time.time()
                    
                avar_theta_deg = update()
                # avar_theta_deg = avar_theta_deg*1.25
                #detectsCollision = max([robot['prox.horizontal'][i] > 1200 for i in range(5)])
                #print('avarage theta deg = ', avar_theta_deg)

                ground_sensors = robot['prox.ground.reflected']
                ground_sensors_max = 1000
                # Adjust these threshold values as needed
                ground_sensors = robot['prox.ground.reflected']
                #print('ground = ',robot['prox.ground.reflected'])
                # Adjust these threshold values as needed
                left_sensor_threshold = 100
                right_sensor_threshold = 100
                direction = random.choice(['left', 'right'])
                if ground_sensors[0] > left_sensor_threshold  and ground_sensors[1]> right_sensor_threshold:
                    # Both sensors detect the line, turn left
                    if direction == 'left':
                        robot['motor.left.target'] = -170
                        robot['motor.right.target'] = 150   
                        time.sleep(0.5) 
                        pass
                    else:
                        robot['motor.left.target'] = 150
                        robot['motor.right.target'] = -170
                        time.sleep(0.5)
                        pass
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
                            robot['motor.left.target'] = 300
                            robot['motor.right.target'] = 20
                            time.sleep(wait)
                        case theta if -30 <= theta < -5:
                            robot["leds.top"] = [0, 255, 255]
                            time.sleep(wait)
                            robot['motor.left.target'] = 200
                            robot['motor.right.target'] = 20
                            time.sleep(wait)
                        case theta if -3 <= theta <= 3:
                            robot["leds.top"] = [255, 255, 255]
                            time.sleep(wait)
                            robot['motor.left.target'] = 50
                            robot['motor.right.target'] = 50
                            time.sleep(0.1)
                            direction = random.choice(['left', 'right'])
                            if direction == 'left':
                                robot['motor.left.target'] = -150
                                robot['motor.right.target'] = 150
                                time.sleep(0.1)
                                pass
                            else:
                                robot['motor.left.target'] = 150
                                robot['motor.right.target'] = -150
                                time.sleep(0.1)
                                pass
                            time.sleep(waiturn)
                        case theta if 5 < theta <= 30:
                            robot["leds.top"] = [255, 255, 0]
                            time.sleep(wait)
                            robot['motor.right.target'] = 200
                            robot['motor.left.target'] = 20
                            time.sleep(wait)
                        case theta if theta > 30:
                            robot["leds.top"] = [255, 0, 0]
                            time.sleep(wait)
                            robot['motor.right.target'] = 300
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
        
        #q_contents = [q.get() for _ in range(q.qsize())]
        #
        ##time.sleep(1)
        #print('q_contents = ', np.shape(q_contents))
        #rec = np.concatenate(q_contents)
        #print('rec = ', np.shape(rec))
        #  
#
        #rec2besaved = rec[:, :channels]
        #save_path = '/home/thymio/robat_py/recordings/'
        #full_path = os.path.join(save_path, args.filename)
        #with sf.SoundFile(full_path, mode='x', samplerate=rec_samplerate,
        #                channels=args.channels, subtype=args.subtype) as file:
        #    file.write(rec2besaved)
        #    print(f'\nsaved to {args.filename}\n')
        sd.stop()

        #print("\nREC START TIME: \n", startime.strftime("%Y-%m-%d %H:%M:%S"))
        #print("\nSTOP REC TIME: \n", stoptime.strftime("%Y-%m-%d %H:%M:%S"))

        #print("fieldrecorder_trigger_robat.py has been terminated \n FILE IN PATH: \n\n" + path ) 
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

# -*- coding: utf-8 -*-
"""
TODO :
    > fix the default output channel number

Based on the AVR Soundmexpro based scripts written with
Holger R. Goerlitz

Usage instructions :

Press F5 to run the script start the sync signal playback, you should now have
camera live feed

    To begin recording press any key

    To stop recording press any key again

    To end the session , either :
        a) Interrupt with a system exit : CTRL+c (creates a tiny file at the end because of keyboard triggering)
        b) Press the interrupt button with the mouse (doesn't create an extra file )
            This only works on the IPython shell

The program records N input channels simultaneously 

By default, even though data is being collected from all channels, only some channels
are saved into the wav file.

Created on Mon Nov 06 18:10:01 2017

 Version 0.0.0 ( semantic versioning number)

@author: Thejasvi Beleyur
"""
import os
import queue
import datetime as dt
import time
import numpy as np
import sounddevice as sd
from scipy import signal
import soundfile
import matplotlib.pyplot as plt

class fieldrecorder_trigger():

    def __init__(self,rec_durn, duty_cycle = None, device_name=None,input_output_chs=(8,8),
                 #target_dir = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/ro-bat_V0/recordings',**kwargs):
                 target_dir = '/robat_py/recordings',**kwargs):
        '''

        Inputs:
        rec_durn : float. duration of the whole session in seconds
        device_name : string. name of the device as shown by sd.query_devices()
                    Defaults to None - which will throw an error if there are not at
                    least 3 output channels

        input_output_chs: tuple with integers. Number of channels for recording and playback.

        target_dir : file path. place where the output WAV files will be saved

        **kwargs:
            exclude_channels: list with integers. These channels will not be saved
                              into the WAV file. Defaults to the digital channels
                              in the double Fireface UC setup
            
            rec_bout : integer. Number of seconds each recording bout should be 
                       Defaults to 10 seconds.
            
            trigger_level : integer <=0. The dB rms for the level at which the recordings
                        get triggered.
            monitor_channels : array like with integers. Channel indices that will be 
                               used for monitoring
            
            bandpass_freqs : tuple. Highpass and lowpass frequencies for the trigger calculation
                       Defaults to the whole frequency spectrum.

        '''
        self.rec_durn = rec_durn
        #self.press_count = 0
        self.recording = False
        #self.sync_freq = 25
        self.device_name = device_name
        self.input_output_chs = input_output_chs
        self.target_dir = target_dir
        #self.FFC_interval = 2
        self.fs = 48000
        self.blocksize = 4096
        #self.duty_cycle = duty_cycle

        
        if self.device_name  is None:
            self.tgt_ind = None
        else:
            self.get_device_indexnumber(self.device_name)

        try:
            expanded_path = os.path.expanduser(target_dir)
            os.chdir(expanded_path)
        except:
            raise ValueError('Unable to find the target directory: ' + target_dir)

        self.all_recchannels = range(self.input_output_chs[0])

        if 'exclude_channels' not in kwargs.keys():
            self.exclude_channels = []
        else:
            self.exclude_channels = kwargs['exclude_channels']

        self.save_channels  = list(set(self.all_recchannels) - set(self.exclude_channels))
        
        ## set the recording bout duration : 
        #if 'rec_bout' not in kwargs.keys():
        #    self.rec_bout = 10
        #else : 
        #    self.rec_bout = kwargs['rec_bout']

        #if 'trigger_level' not in kwargs.keys():
        #    self.trigger_level = -50 # dB level ref max
        #else:
        #    self.trigger_level = kwargs['trigger_level']
        
        if 'monitor_channels' not in kwargs.keys():
            self.monitor_channels = [0,1,2,3]
        else:
            self.monitor_channels = kwargs['monitor_channels']
        
        if 'bandpass_freqs' in kwargs.keys():
            self.highpass_freq, self.lowpass_freq = kwargs['bandpass_freqs']
            nyq_freq = self.fs/2.0
            self.b, self.a = signal.butter(4, [self.highpass_freq/nyq_freq,
                                               self.lowpass_freq/nyq_freq],
                                btype='bandpass')
            self.bandpass = True
        else:
            self.bandpass = False
            
        #if duty_cycle is None:
        #    self.minimum_interval = 0
        #else:
        #    self.minimum_interval = ((1-duty_cycle)/duty_cycle)*self.rec_bout
            

    def thermoacousticpy(self):
        '''
        Performs the synchronised recording of thermal cameras and audio.

        '''

        self.S = sd.InputStream(samplerate=self.fs,blocksize=self.blocksize,
                           channels=5,device=self.tgt_ind)

        start_time = np.copy(self.S.time)
        rec_time = np.copy(self.S.time)
        end_time =  start_time + self.rec_durn


        self.q = queue.Queue()

        self.S.start()
        num_recordings = 0
        
        try:

#            while rec_time < end_time:
#                
            self.mic_inputs = self.S.read(self.blocksize)
            self.ref_channels = self.mic_inputs[0][:,self.monitor_channels]
            self.ref_channels_bp = self.bandpass_sound(self.ref_channels)
            self.start_recording = True
#                print('start_rec = ', self.start_recording)
            print('starting_recording')     
            while self.start_recording:
#                
#                    self.recbout_start_time = np.copy(self.S.time)
#                    self.recbout_end_time = self.recbout_start_time + self.rec_bout
#                i = 0
        #                    print(self.S.time)
        #                    while  self.recbout_end_time >= self.S.time:   
#                if i != 0:
                self.q.put(self.S.read(self.blocksize))
#                else:
#                    self.q.put(self.mic_inputs)
#                i += 1
#                        
#                    print(self.S.time)    
#
#                    self.start_recording = False 
#                      
#
#            num_recordings += 1 
#                    prev_rectime = np.copy(self.S.time)

        except (KeyboardInterrupt, SystemExit):
            self.start_recording = False
            self.empty_qcontentsintolist()
            self.save_qcontents_aswav()
            print('\nStopping recording ..exiting ')

        self.S.stop()
        print('Queue size is',self.q.qsize())
        return(self.fs)

    def bandpass_sound(self, rec_buffer):
        """
        """
        if self.bandpass:
            rec_buffer_bp = np.apply_along_axis(lambda X : signal.lfilter(self.b, self.a, X),
                                                0, rec_buffer)
            return(rec_buffer_bp)
        else:
            return(rec_buffer)
        
    def empty_qcontentsintolist(self):
        try:
            self.q_contents = [ self.q.get()[0] for i in range(self.q.qsize()) ]
        except:
            raise IOError('Unable to empty queue object contents')

        pass

    def save_qcontents_aswav(self):

        print('Saving file now...')

        self.rec = np.concatenate(self.q_contents)

        self.rec2besaved = self.rec[:,self.save_channels]

        timenow = dt.datetime.now()
        self.timestamp = timenow.strftime('%Y-%m-%d_%H-%M-%S')
        self.idnumber =  int(time.mktime(timenow.timetuple())) #the unix time which provides a 10 digit unique identifier

        main_filename = 'MULTIWAV_' + self.timestamp+'_'+str(self.idnumber) +'.WAV'

        try:
            print('trying to save file... ')

            soundfile.write(main_filename,self.rec2besaved,self.fs)

            print('File saved')

            pass

        except:
            raise IOError('Could not save file !!')


        pass

    def get_device_indexnumber(self,device_name): 
        '''
        Check for the device name in all of the recognised devices and
        return the index number within the list.

        '''
        self.device_list = sd.query_devices()

        self.tgt_dev_name = device_name
        self.tgt_dev_bool = [self.tgt_dev_name in each_device['name'] for each_device in self.device_list]

        if not True in self.tgt_dev_bool:

            print (sd.query_devices())

            raise ValueError('The input device \n' + self.tgt_dev_name+
            '\n could not be found, please look at the list above'+
                             ' for all the recognised devices'+
                             ' \n Please use sd.query_devices to check the  recognised'
                             +' devices on this computer')

        if sum(self.tgt_dev_bool) > 1 :
           raise ValueError('Multiple devices with the same string found'
           + ' please enter a more specific device name'
           ' \n Please use sd.query_devices to check the recognised'+
           ' devices on this computer')

        else:
            self.tgt_ind = int(np.argmax(np.array(self.tgt_dev_bool)))


if __name__ == '__main__':

    dev_name = 'MCHStreamer'
    in_out_channels = (5,5)
#    if not os.path.exists('/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/recordings'):
#            os.makedirs('/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/recordings')
#    tgt_directory = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/recordings'
    if not os.path.exists('/home/thymio/robat_py/recordings'):
        os.makedirs('/home/thymio/robat_py/recordings')
    tgt_directory = '/home/thymio/robat_py/recordings'
    a = fieldrecorder_trigger(3500, input_output_chs= in_out_channels,
                              device_name= dev_name, target_dir= tgt_directory,
                              trigger_level=-1.0, duty_cycle=0.18,
                              monitor_channels=[0,1,4], rec_bout = 5.0,
                              bandpass_freqs = [20.0, 20000.0]
                              )
    fs = a.thermoacousticpy()



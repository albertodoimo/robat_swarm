# -*- coding: utf-8 -*-
"""
Short example to show how to test ADC-DAC latency with sounddevice
==================================================================



Notes
-----
This example uses a queue object to hold audio buffers, and unless you run the entire code chunk, 
the queue object will be empty and the results spectrogr

Created on Wed May 15 06:29:04 2024

@author: theja
"""

import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal
import queue
import time

tone_durn = 10e-3 # seconds
fs = 48000
t_tone = np.linspace(0, tone_durn, int(fs*tone_durn))
chirp = signal.chirp(t_tone, 20e3, t_tone[-1], 2e2)
chirp *= signal.windows.hann(chirp.size)
output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.02)))))

plt.figure(figsize=(10, 6))
t_audio = np.linspace(0, output_chirp.shape[0]/fs, output_chirp.shape[0])
plt.plot(t_audio, output_chirp)   
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('output.png')
plt.show()

output_tone = np.float32(np.reshape(output_chirp, (-1, 1)))

# device IDs, you'll need to check the device ID with sd.query_devices()
# and give the input/output device numbers. 

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())

def callback_sine(indata, outdata, frames, time, status):
    outdata[:] = output_tone
    all_input_data.put(indata.copy())

# create a stream, and set the blocksize to match that of the output signal
stream_inout = sd.Stream(samplerate = fs,
                         blocksize = output_chirp.shape[0],
                         device = (usb_fireface_index, usb_fireface_index),
                         channels = (7, 2),
                         callback = callback_sine 
                         )
# time.sleep(0.5)
# start the stream 
all_input_data = queue.Queue() # Creates a queue object to store input data
start_time = time.time()
try:    
    with stream_inout:
        while (time.time() - start_time) < 0.5:
            pass
    stream_inout.stop()
except :
    print('Exception occurred')

# load the input audio 
all_input_audio = []
while not all_input_data.empty():
    all_input_audio.append(all_input_data.get())            
input_audio = np.concatenate(all_input_audio)

plt.figure(figsize=(12, 6))
aa = plt.subplot(211) 
plt.specgram(input_audio[:, 0], Fs=fs, NFFT=128, noverlap=64)    
plt.xticks(np.arange(0, t_audio[-1], 0.01), fontsize = 8, rotation=90)
plt.subplot(212, sharex=aa)
t_audio = np.linspace(0, input_audio.shape[0]/fs, input_audio.shape[0])
print('time length sec:', t_audio[-1])
plt.plot(t_audio, input_audio[:, 0])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Input Audio Signal')
plt.tight_layout()
plt.xticks(np.arange(0, t_audio[-1], 0.01), fontsize = 8, rotation=90)
plt.grid()
plt.savefig('latency.png')
plt.show()
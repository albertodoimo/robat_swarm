# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:44:27 2024

@author: theja
"""

import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt 
import sounddevice as sd
import soundfile as sf

import time

#%%
# make a sweep
# durns = np.array([3, 4, 5, 8, 10] )*1e-3
durns = np.array([4*1e-3])
fs = 48000 # Hz

all_sweeps = []
for durn in durns:
    time.sleep(0.1)  # wait for the sound card to settle
    t = np.linspace(0, durn, int(fs*durn))
    start_f, end_f = 100, 24e3
    sweep = signal.chirp(t, start_f, t[-1], end_f)
    sweep *= signal.windows.tukey(sweep.size, 0.2)
    sweep *= 0.8
    sweep_padded = np.pad(sweep, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])
    all_sweeps.append(sweep_padded)
    time.sleep(0.1)  # wait for the sound card to settle
    
sweeps_combined = np.concatenate(all_sweeps)
sf.write('01_24k_1sweeps_4ms_amp08_48k.wav', sweeps_combined, samplerate=fs)
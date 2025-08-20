# Output of a specified file from the MCHStreamer Card on the robat for output loudspeaker testing

# Author: Alberto 
# Date: 11-6-2025

import sounddevice as sd
import soundfile as sf
import time 
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import time


def get_card(device_list):
    """
    Get the index of the ASIO card in the device list.
    Parameters:
    - device_list: list of devices (usually = sd.query_devices())

    Returns: index of the card in the device list
    """
    for i, each in enumerate(device_list):
        dev_name = each['name']
        name = 'MCHStreamer' in dev_name
        if name:
            return i
    return None

def read_audiofile(filepath):
    audio, fs = sf.read(filepath)
    return audio, fs

if __name__ == "__main__":

    print(sd.query_devices())
    sd.default.device = get_card(sd.query_devices())
    print('selected device:\n', sd.default.device)
    
    # Path to the playback file
    pbkfile_path = './robat_py/01_24k_1sweeps_4ms_amp08_48k.wav'
    amps = [0.1, 0.3, 0.5, 0.7, 1]  
    pbk, fs_or = read_audiofile(pbkfile_path)
    print('Playback file loaded:', pbkfile_path)
    print('Sample rate:', fs_or)

    # Resampling the playback signal to 48kHz if necessary
    fs = 48000
    if fs_or != fs:
        pbk = signal.resample(pbk, int(len(pbk) * fs / fs_or))
        print(f'Resampled playback signal to {fs} Hz.')
    
    # Play the audio file
    print('\nPlaying back the audio file...')
    # for amp in amps:
    #     sd.play(amp*pbk, samplerate=fs, blocking=True) 
    #     time.sleep(1)
    
    try:
        while True:
            sd.play(1*pbk, samplerate=fs, blocking=True) 
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('\nPlayback stopped')
        sd.stop()

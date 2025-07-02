import numpy as np
from functions.gcc_phat import gcc_phat

def calc_multich_delays(multich_audio,ref_sig,fs, ref_channel):
    '''
    Calculates peak delay based with reference of ref channel
    Parameters:
    - multich_audio: multichannel audio signal
    - ref_sig: reference signal
    - fs: sampling frequency
    - ref_channel: reference channel
    
    Returns:
    - delay_set: array of delays
    '''
    nchannels = multich_audio.shape[1]
    delay_set = []
    i=0
    while i < nchannels:
        if i != ref_channel:
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
import numpy as np
import sounddevice as sd
from functions.calc_dBrms import calc_dBrms

def check_if_above_level(mic_inputs,trigger_level,critical_level):
    """Checks if the dB rms level of the input recording buffer is above
    threshold. If any of the microphones are above the given level then 
    recording is initiated. 
    
    Parameters:
    - mic_inputs : Nsamples x Nchannels np.array. Data from soundcard
    - trigger_level :  If the input data buffer has a dB rms >= this value then True will be returned. 
    - max_dBrms = np.max(dBrms_channel) max dBrms value across all channels.
                
    Returns:
        
        trigger_bool : Boolean. True if the buffer dB rms is >= the trigger_level
        critical_bool : Boolean. True if the buffer dB rms is >= the critical_level

    """ 

    dBrms_channel = np.apply_along_axis(calc_dBrms, 0, mic_inputs)   
    #print('dBrms_channel=',dBrms_channel)     
    trigger_bool = np.any( dBrms_channel >= trigger_level)
    critical_bool = np.any( dBrms_channel >= critical_level)
    if critical_bool:
        trigger_bool = False
    max_dBrms = np.max(dBrms_channel)
    #print('max dBrms =',max_dBrms)
    #print('above level =',above_level)
    
    return(trigger_bool,critical_bool,dBrms_channel)

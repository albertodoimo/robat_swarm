import numpy as np

def calc_dBrms(one_channel_buffer):
    """
    Calculate the dB RMS of a single channel buffer.
    Parameters:
    - one_channel_buffer : 1D np.array. Single channel buffer.

    Returns:
    - dB_rms : float. dB RMS value of the buffer.
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
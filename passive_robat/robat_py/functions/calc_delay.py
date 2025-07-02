import numpy as np

def calc_delay(two_ch,fs):
    """
    Parameters:
    - two_ch : (Nsamples, 2) np.array
        Input audio buffer
    fs : Frequency of sampling in Hz.

    Returns:
    - delay : float
        The time-delay in seconds between the arriving audio across the 
        channels. 
    """
    for each_column in range(2):
        two_ch[:,each_column] = two_ch[:,each_column]

    cc = np.correlate(two_ch[:,0],two_ch[:,1],'same')
    midpoint = cc.size/2.0
    delay = np.argmax(cc) - midpoint
    # convert delay to seconds
    delay *= 1/float(fs)
    return delay
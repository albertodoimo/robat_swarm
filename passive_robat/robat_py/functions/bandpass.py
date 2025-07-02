import numpy as np
from scipy import signal

def bandpass(rec_buffer,highpass_freq,lowpass_freq,fs):
    """
    Applies a bandpass filter to the input buffer.
    Parameters:
    - rec_buffer: input buffer
    - highpass_freq: highpass frequency
    - lowpass_freq: lowpass frequency
    - fs: sampling frequency

    Returns:
    - bandpass filtered buffer
    """
    nyq_freq = fs/2.0
    b, a = signal.butter(4, [highpass_freq/nyq_freq,lowpass_freq/nyq_freq],btype='bandpass') # to be 'allowed' in Hz.
    rec_buffer_bp = np.apply_along_axis(lambda X : signal.lfilter(b, a, X),0, rec_buffer)
    return(rec_buffer_bp)

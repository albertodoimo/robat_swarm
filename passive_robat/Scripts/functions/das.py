import numpy as np
from scipy.signal import stft

def das_filter_v2(y, fs, nch, d, bw, theta=np.linspace(-90, 90, 36), c=343):    
    """
    Simple multiband Delay-and-Sum spatial filter implementation.
    Parameters:
    - y: mic array signals
    - fs: sampling rate
    - nch: number of mics in the array
    - d: mic spacing
    - bw: (low freq, high freq)
    - theta: angle vector
    - c: sound speed

    Returns: normalized average of the spatial energy distribution estimation across bands
    """
    win_len = 256
    f_spec_axis, _, spectrum = stft(y, fs=fs, window=np.ones((win_len, )), nperseg=win_len, noverlap=win_len-1, axis=0)
    bands = f_spec_axis[(f_spec_axis > bw[0]) & (f_spec_axis < bw[1])]
    bands = np.array([bands[0], bands[len(bands)//3], bands[2*len(bands)//3], bands[-1]])
    # bands = np.array([bands[len(bands)//2]])
    p = np.zeros_like(theta, dtype=complex)
    
    for f_c in bands:
        w_s = (2*np.pi*f_c*d*np.sin(np.deg2rad(theta))/c)        
        a = np.exp(np.outer(np.linspace(0, nch-1, nch), -1j*w_s))     
        spec = spectrum[f_spec_axis == f_c, :, :].squeeze()
        cov_est = np.cov(spec, bias=True)
        
        for i, _ in enumerate(theta):
          p[i] += a[:, i].T.conj() @ cov_est @ a[:, i]/(nch**2)
    
    mag_p = np.abs(p)/len(bands)
        
    return theta, mag_p
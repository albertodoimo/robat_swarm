# -*- coding: utf-8 -*-
"""
SPL level from a non-calibration microphone
===========================================
See README for broad experimental workflow. 


Created April 2025
@author: thejasvi
"""
import numpy as np 
import scipy.signal as signal
import scipy 
from scipy.interpolate import interp1d
import time

def pascal_to_dbspl(X):
    '''
    Converts Pascals to dB SPL re 20 uPa
    '''
    return dB(X/20e-6)

def rms(X):
    return np.sqrt(np.mean(X**2))

dB = lambda X: 20*np.log10(abs(np.array(X).flatten()))

db_to_linear = lambda X: 10**(X/20)


# bin width clarification https://stackoverflow.com/questions/10754549/fft-bin-width-clarification
def get_rms_from_fft(freqs, spectrum, **kwargs):
    '''Use Parseval's theorem to get the RMS level of each frequency component
    This only works for RFFT spectrums!!!
    
    Parameters
    ----------
    freqs : (Nfreqs,) np.array >0 values
    spectrum : (Nfreqs,) np.array (complex)
    freq_range : (2,) array-like
        Min and max values
    
    Returns 
    -------
    root_mean_squared : float
        The RMS of the signal within the min-max frequency range
   
    '''
    minfreq, maxfreq = kwargs['freq_range']
    relevant_freqs = np.logical_and(freqs>=minfreq, freqs<=maxfreq)
    spectrum_copy = spectrum.copy()
    spectrum_copy[~relevant_freqs] = 0
    mean_sigsquared = np.sum(abs(spectrum_copy)**2)/spectrum.size
    root_mean_squared = np.sqrt(mean_sigsquared/(2*spectrum.size-1))
    return root_mean_squared


def calc_native_freqwise_rms(X, fs):
    '''
    Converts the FFT spectrum into a band-wise rms output. 
    The frequency-resolution of the spectrum/audio size decides
    the frequency resolution in general. 
    
    Parameters
    ----------
    X : np.array
        Audio
    fs : int
        Sampling rate in Hz
    
    Returns 
    -------
    fftfreqs, freqwise_rms : np.array
        fftfreqs holds the frequency bins from the RFFT
        freqwise_rms is the RMS value of each frequency bin. 
    '''
    time1 = time.time()
    rfft = np.fft.rfft(X)
    fftfreqs = np.fft.rfftfreq(X.size, 1/fs)
    # now calculate the rms per frequency-band
    # print('RFFT computation time:', time.time() - time1)
    freqwise_rms = []

    abs_rfft_squared = np.abs(rfft)**2
    mean_sq_freq = abs_rfft_squared / rfft.size
    rms_freq = np.sqrt(mean_sq_freq / (2*rfft.size-1))
    freqwise_rms = rms_freq.tolist()

    # freqwise_rms2 = []
    # for each in rfft:
    #     mean_sq_freq2 = np.sum(abs(each)**2)/rfft.size
    #     rms_freq2 = np.sqrt(mean_sq_freq2/(2*rfft.size-1))
    #     freqwise_rms2.append(rms_freq2)
    return fftfreqs, freqwise_rms

    

# Make an interpolation function 
def interpolate_freq_response(mic_freq_response, new_freqs):
    ''' 
    Parameters
    ----------
    mic_freq_response : tuple/list
        A tuple/list with two entries: (centrefreqs, centrefreq_RMS).
        
    new_freqs : list/array-like
        A set of new centre frequencies that need to be interpolated to. 

    Returns 
    -------
    tgtmicsens_interp : 
        
    Attention
    ---------
    Any frequencies outside of the calibration range will automatically be 
    assigned to the lowest sensitivity values measured in the input centrefreqs
    
    '''
    centrefreqs, mic_sensitivity = mic_freq_response 
    tgtmic_sens_interpfn = interp1d(centrefreqs, mic_sensitivity,
                                    kind='cubic', bounds_error=False,
                                    fill_value=np.min(mic_sensitivity))
    # interpolate the sensitivity of the mic to intermediate freqs
    tgtmicsens_interp = tgtmic_sens_interpfn(new_freqs)
    return tgtmicsens_interp
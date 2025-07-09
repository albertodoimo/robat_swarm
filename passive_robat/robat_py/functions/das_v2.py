import numpy as np
from scipy.signal import stft, ShortTimeFFT
import time
from matplotlib import pyplot as plt
import librosa 

def das_filter(y, fs, nch, d, bw, theta, c=343, wlen=64, show=False):    
  """
  Simple multiband Delay and Sum spatial filter implementation.

  Parameters:

    y: mic array signals

    fs: sampling rate

    nch: number of mics in the array

    d: mic spacing

    bw: (low freq, high freq)

    theta: angle vector

    c: sound speed

    wlen: window length for stft

    show: plot the pseudospectrum for each band

  Returns: 
    
    theta: angle axis
    
    mag_p: magnitude of average spatial energy distribution estimation across bands
  """

  time1 = time.time()

  f_spec_axis, _, spectrum = stft(y, fs=fs, window=np.ones((wlen, )), nperseg=wlen, noverlap=wlen-1, axis=0)
  bands = f_spec_axis[(f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1])]
  p = np.zeros_like(theta, dtype=complex)
  p_i = np.zeros_like(theta, dtype=complex)

  if show:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
  
  time2 = time.time()
  print('STFT computation time:', time2 - time1)
  for f_c in bands:
      w_s = (2*np.pi*f_c*d*np.sin(np.deg2rad(theta))/c)        
      a = np.exp(np.outer(np.linspace(0, nch-1, nch), -1j*w_s))
      a_H = a.T.conj()     
      spec = spectrum[f_spec_axis == f_c, :, :].squeeze()
      cov_est = np.cov(spec, bias=True)
      
      for i in range(len(theta)):        
        p_i[i] = a_H[i, :] @ cov_est @ a[:, i]/(nch**2)
      
      p += p_i
      
      if show:
        ax.plot(np.deg2rad(theta), 10*np.log10(np.abs(p_i)), label=f'{f_c} Hz')
  
  if show:
    ax.set_xlim((-np.pi/2, np.pi/2))
    plt.ylabel('Magnitude (dB)')
    ax.set_title('Pseudospectra')
    ax.set_theta_offset(np.pi/2)
    plt.legend()
    plt.savefig('das1_pseudospectra.png')
    plt.show()
    
  mag_p = np.abs(p)/len(bands)
  print('DAS computation time:', time.time() - time2)
  return theta, mag_p, f_spec_axis, spectrum, bands


import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import time

def fast_das_numpy(y, fs, nch, d, bw, theta, c=343, wlen=64, show=False):
    time1 = time.time()

    # -- Step 1: STFT using NumPy only --
    y = np.asarray(y)
    frames = sliding_window_view(y, window_shape=(wlen,), axis=0)  # shape: (n_frames, nch, wlen)
    frames = frames.transpose(0, 2, 1)  # shape: (n_frames, wlen, nch)
    spectrum = np.fft.rfft(frames, axis=1)  # (n_frames, n_freqs, nch)

    n_freqs = spectrum.shape[1]
    f_spec_axis = np.fft.rfftfreq(wlen, d=1/fs)  # frequencies corresponding to FFT bins

    bands = f_spec_axis[(f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1])]
    band_idxs = np.where((f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1]))[0]

    p = np.zeros_like(theta, dtype=complex)
    p_i = np.zeros_like(theta, dtype=complex)

    if show:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    time2 = time.time()
    print('STFT computation time:', time2 - time1)

    for idx in band_idxs:
        f_c = f_spec_axis[idx]

        # Steering vector for all angles
        w_s = 2 * np.pi * f_c * d * np.sin(np.deg2rad(theta)) / c
        a = np.exp(np.outer(np.arange(nch), -1j * w_s))  # (nch, len(theta))
        a_H = np.conj(a.T)  # (len(theta), nch)

        spec = spectrum[:, idx, :]  # (n_frames, nch)
        cov_est = np.cov(spec.T, bias=True)  # (nch, nch)

        for i in range(len(theta)):
            p_i[i] = a_H[i, :] @ cov_est @ a[:, i] / (nch ** 2)

        p += p_i

        if show:
            ax.plot(np.deg2rad(theta), 10 * np.log10(np.abs(p_i)), label=f'{f_c:.1f} Hz')

    if show:
        ax.set_xlim((-np.pi/2, np.pi/2))
        plt.ylabel('Magnitude (dB)')
        ax.set_title('Pseudospectra')
        ax.set_theta_offset(np.pi/2)
        ax.legend()
        plt.savefig('das2_pseudospectra.png')
        plt.show()

    mag_p = np.abs(p) / len(bands)
    print('DAS computation time:', time.time() - time2)

    return theta, mag_p, f_spec_axis, spectrum, f_spec_axis[band_idxs]

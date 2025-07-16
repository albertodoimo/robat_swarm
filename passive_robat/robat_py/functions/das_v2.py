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
  
  # print('STFT computation time:', time2 - time1)

  # speed version of DAS 
  band_idxs = np.where((f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1]))[0]
  nch_range = np.linspace(nch-1, 0, nch)  # np.linspace( nch-1,0, nch) = -90>0>90; np.linspace( 0,nch-1, nch) = 90>0>-90
  theta_rad = np.deg2rad(theta)
  w_s_mat = 2 * np.pi * bands[:, None] * d * np.sin(theta_rad) / c  # shape: (n_bands, len(theta))
  a_mat = np.exp(np.outer(nch_range, -1j * w_s_mat.ravel())).reshape(nch, len(bands), len(theta))  # (nch, n_bands, len(theta))
  a_H_mat = np.conj(a_mat.transpose(1, 2, 0))  # (n_bands, len(theta), nch)

  for b_idx, f_idx in enumerate(band_idxs):
      spec = spectrum[f_idx, :, :]  # (nch, n_frames)
      cov_est = np.cov(spec, bias=True)
      a = a_mat[:, b_idx, :]  # (nch, len(theta))
      a_H = a_H_mat[b_idx, :, :]  # (len(theta), nch)
      p_i = np.einsum('ij,jk,ki->i', a_H, cov_est, a) / (nch ** 2)
      p += p_i

      if show:
          ax.plot(theta_rad, 10 * np.log10(np.abs(p_i)), label=f'{bands[b_idx]:.1f} Hz')
  # print('DAS computation time 1 :', time.time() - time2)
  mag_p = np.abs(p)/len(bands)

  # # normal version 
  # for f_c in bands:
  #     w_s = (2*np.pi*f_c*d*np.sin(np.deg2rad(theta))/c)        
  #     a = np.exp(np.outer(np.linspace(nch-1,0, nch), -1j*w_s)) # np.linspace( nch-1,0, nch) = -90>0>90; np.linspace( 0,nch-1, nch) = 90>0>-90
  #     a_H = a.T.conj()     
  #     spec = spectrum[f_spec_axis == f_c, :, :].squeeze()
  #     cov_est = np.cov(spec, bias=True)
      
  #     for i in range(len(theta)):        
  #       p_i[i] = a_H[i, :] @ cov_est @ a[:, i]/(nch**2)
      
  #     p += p_i
      
  #     if show:
  #       ax.plot(np.deg2rad(theta), 10*np.log10(np.abs(p_i)), label=f'{f_c} Hz')
  # print('DAS computation time 2 :', time.time() - time2)
  # mag_p = np.abs(p)/len(bands)

  if show:
    ax.set_xlim((-np.pi/2, np.pi/2))
    plt.ylabel('Magnitude (dB)')
    ax.set_title('Pseudospectra')
    ax.set_theta_offset(np.pi/2)
    plt.legend()
    plt.savefig('das1_pseudospectra.png')
    plt.show()
    
  return theta, mag_p, f_spec_axis, spectrum, bands


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def pow_two_pad_and_window(vec, fs, show = True):
    """
    Pad a vector with zeros to the next power of two and apply a Tukey window.
    Parameters:
    - vec: input vector
    - fs: sampling rate
    - show: plot the windowed vector and its spectrogram

    Returns:
    - padded_windowed_vec: padded and windowed vector
    """
    window = signal.windows.tukey(len(vec), alpha=0.2)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(windowed_vec))
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, windowed_vec)
        plt.subplot(2, 1, 2)
        plt.specgram(windowed_vec, NFFT=256, Fs=fs)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)
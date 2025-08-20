import numpy as np
from scipy.fftpack import fft

def gcc_phat(sig,refsig, fs):
    """
    Computes the cross-correlation between the two signals.
    Parameters:
    - sig: the first signal 
    - refsig: the second signal chosen as reference
    - fs: the sampling frequency

    Returns:
    - the time delay between the two signals
    """
    n = sig.shape[0] + refsig.shape[0]
    SIG = fft(sig, n=n)
    REFSIG = fft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.ifft(R / np.abs(R))
    max_shift = int(np.floor(n / 2))
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    #plt.plot(cc)
    #plt.show()
    #plt.title('gcc-phat')
    shift = np.argmax(np.abs(cc)) - max_shift #

    return -shift / float(fs) # time delay

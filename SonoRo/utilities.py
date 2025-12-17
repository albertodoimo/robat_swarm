#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created: 2025-12-16
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description
Utility functions used to run SonoRo_swarm.py

"""

import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.signal import stft
import time
import matplotlib.pyplot as plt

# Audio processing functions


def pascal_to_dbspl(X):
    """Converts Pascals to dB SPL re 20 uPa"""
    return dB(X / 20e-6)


def rms(X):
    """Calculates the root mean square of the input signal X

    Parameters
    ----------
    X : np.array
        Input signal

    Returns
    -------
    rms_value :
        The RMS value of the input signal

    """
    return np.sqrt(np.mean(X**2))


def calc_dBrms(one_channel_buffer):
    """Calculate the dB RMS of a single channel buffer.

    Parameters
    -----------
    one_channel_buffer : 1D np.array.
        Single channel buffer.

    Returns
    -----------
    dB_rms : float.
        dB RMS value of the buffer.

    """
    squared = np.square(one_channel_buffer)
    mean_squared = np.mean(squared)
    root_mean_squared = np.sqrt(mean_squared)
    # print('rms',root_mean_squared)
    try:
        dB_rms = 20.0 * np.log10(root_mean_squared)
    except ValueError:
        dB_rms = -np.inf
    return dB_rms


dB = lambda X: 20 * np.log10(abs(np.array(X).flatten()))

db_to_linear = lambda X: 10 ** (X / 20)


# bin width clarification https://stackoverflow.com/questions/10754549/fft-bin-width-clarification
def get_rms_from_fft(freqs, spectrum, **kwargs):
    """Use Parseval's theorem to get the RMS level of each frequency component
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

    """
    minfreq, maxfreq = kwargs["freq_range"]
    relevant_freqs = np.logical_and(freqs >= minfreq, freqs <= maxfreq)
    spectrum_copy = spectrum.copy()
    spectrum_copy[~relevant_freqs] = 0
    mean_sigsquared = np.sum(abs(spectrum_copy) ** 2) / spectrum.size
    root_mean_squared = np.sqrt(mean_sigsquared / (2 * spectrum.size - 1))
    return root_mean_squared


def calc_native_freqwise_rms(X, fs):
    """Converts the FFT spectrum into a band-wise rms output.
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

    """
    time1 = time.time()
    rfft = np.fft.rfft(X)
    fftfreqs = np.fft.rfftfreq(X.size, 1 / fs)
    # now calculate the rms per frequency-band
    freqwise_rms = []

    abs_rfft_squared = np.abs(rfft) ** 2
    mean_sq_freq = abs_rfft_squared / rfft.size
    rms_freq = np.sqrt(mean_sq_freq / (2 * rfft.size - 1))
    freqwise_rms = rms_freq.tolist()

    return fftfreqs, freqwise_rms


def interpolate_freq_response(mic_freq_response, new_freqs):
    """
    Parameters
    ----------
    mic_freq_response : tuple/list
        A tuple/list with two entries: (centrefreqs, centrefreq_RMS).

    new_freqs : list/array-like
        A set of new centre frequencies that need to be interpolated to.

    Returns
    -------
    tgtmicsens_interp :

    Note
    ---------
    Any frequencies outside of the calibration range will automatically be
    assigned to the lowest sensitivity values measured in the input centrefreqs

    """
    centrefreqs, mic_sensitivity = mic_freq_response
    tgtmic_sens_interpfn = interp1d(
        centrefreqs,
        mic_sensitivity,
        kind="cubic",
        bounds_error=False,
        fill_value=np.min(mic_sensitivity),
    )
    # interpolate the sensitivity of the mic to intermediate freqs
    tgtmicsens_interp = tgtmic_sens_interpfn(new_freqs)
    return tgtmicsens_interp


def bandpass(rec_buffer, highpass_freq, lowpass_freq, fs):
    """Applies a bandpass filter to the input buffer.

    Parameters
    ----------
    rec_buffer : ndarray
        Input buffer to be filtered.
    highpass_freq : float
        Highpass frequency in Hz.
    lowpass_freq : float
        Lowpass frequency in Hz.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    rec_buffer_bp : ndarray
        Bandpass filtered buffer with the same shape as the input.

    """
    nyq_freq = fs / 2.0
    b, a = signal.butter(
        4, [highpass_freq / nyq_freq, lowpass_freq / nyq_freq], btype="bandpass"
    )  # to be 'allowed' in Hz.
    rec_buffer_bp = np.apply_along_axis(
        lambda X: signal.lfilter(b, a, X), 0, rec_buffer
    )
    return rec_buffer_bp


def matched_filter(recording, chirp_template):
    """Apply a matched filter to the signal using the provided template.

    Parameters
    -----------
    recording : np.ndarray
        The audio signal to be filtered.
    chirp_template : np.ndarray
        The chirp template used for filtering.

    Returns
    --------
    np.ndarray
        The filtered output, which is the envelope of the matched filter response.

    """
    # Apply correlation and shift
    filtered_output = np.roll(
        signal.correlate(recording, chirp_template, "same", method="direct"),
        -len(chirp_template) // 2,
    )
    # Apply Tukey window
    filtered_output *= signal.windows.tukey(filtered_output.size, 0.1)
    # Get the envelope using Hilbert transform
    filtered_envelope = np.abs(signal.hilbert(filtered_output))

    return filtered_envelope


def detect_peaks(filtered_output, sample_rate, prominence, distance):
    """Detect peaks in the matched filter output.

    Parameters
    ----------
    filtered_output : np.ndarray
        The output of the matched filter.
    sample_rate : int
        The sample rate of the audio signal.
    prominence : float
        Minimum prominence of peaks to be detected.
    distance : float
        Minimum distance between peaks in seconds.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks in the filtered output.

    """
    peaks, properties = signal.find_peaks(
        filtered_output, prominence=prominence, distance=distance * sample_rate
    )
    return peaks


def pow_two_pad_and_window(vec, fs, show=True):
    """Pad a vector with zeros to the next power of two and apply a Tukey window.

    Parameters
    ----------
    vec : np.ndarray
        Input vector
    fs : float
        Sampling rate in Hz
    show : bool, optional
        If True, plot the windowed vector and its spectrogram (default: True)

    Returns
    -------
    padded_windowed_vec : np.ndarray
        Padded and windowed vector normalized by its maximum value

    """
    window = signal.windows.tukey(len(vec), alpha=0.2)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(
        windowed_vec,
        (0, 2 ** int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)),
    )

    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(windowed_vec))
        plt.figure()
        plt.title("Windowed Vector and Spectrogram")
        plt.subplot(2, 1, 1)
        plt.plot(t, windowed_vec)
        plt.subplot(2, 1, 2)
        plt.specgram(windowed_vec, NFFT=256, Fs=fs)
        plt.show()

    return padded_windowed_vec / max(padded_windowed_vec)


# Direction of Arrival estimation functions


def avar_angle(delay_set, nchannels, mic_spacing, ref_channel):
    """Calculate the mean angle of arrival to the array with respect to reference channel.

    Parameters
    ----------
    delay_set : array-like
        The time delay between signals
    nchannels : int
        Number of mics in the array
    mic_spacing : float
        Inter-distance between the mics
    ref_channel : int
        Reference channel

    Returns
    -------
    avar_theta : float
        The mean angle of arrival to the array in radians

    """
    theta = []
    if ref_channel != 0:  # centered reference that works with odd mics
        for each in range(0, nchannels // 2):
            theta.append(
                -np.arcsin(
                    (delay_set[each] * 343) / ((nchannels // 2 - each) * mic_spacing)
                )
            )  # rad
            i = nchannels // 2 - each
        for each in range(nchannels // 2, nchannels - 1):
            theta.append(
                np.arcsin((delay_set[each] * 343) / ((i) * mic_spacing))
            )  # rad
            i += 1
    else:
        for each in range(0, nchannels - 1):
            theta.append(
                np.arcsin((delay_set[each] * 343) / ((each + 1) * mic_spacing))
            )  # rad
    avar_theta = np.mean(theta)

    return avar_theta


def calc_delay(two_ch, fs):
    """Calculate the delay between two channels using cross-correlation.

    Parameters
    ----------
    two_ch : (Nsamples, 2) np.array
        Input audio buffer with 2 channels
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    delay : float
        The time-delay in seconds between the arriving audio across the
        channels

    """
    for each_column in range(2):
        two_ch[:, each_column] = two_ch[:, each_column]

    cc = np.correlate(two_ch[:, 0], two_ch[:, 1], "same")
    midpoint = cc.size / 2.0
    delay = np.argmax(cc) - midpoint
    # convert delay to seconds
    delay *= 1 / float(fs)
    return delay


def calc_multich_delays(multich_audio, ref_sig, fs, ref_channel):
    """Calculate delays between channels using GCC-PHAT with a reference signal.

    Parameters
    ----------
    multich_audio : (Nsamples, Nchannels) np.ndarray
        Multichannel audio signal
    ref_sig : (Nsamples,) np.ndarray
        Reference signal
    fs : float
        Sampling frequency in Hz
    ref_channel : int
        Reference channel index

    Returns
    -------
    delay_set : np.ndarray
        Array of time delays in seconds for each channel relative to reference

    """
    nchannels = multich_audio.shape[1]
    delay_set = []
    i = 0
    while i < nchannels:
        if i != ref_channel:
            delay_set.append(
                gcc_phat(multich_audio[:, i], ref_sig, fs)
            )  # gcc phat correlation
            i += 1
        else:
            i += 1
            pass

    return np.array(delay_set)


def check_if_above_level(mic_inputs, trigger_level, critical_level):
    """Checks if the dB rms level of the input recording buffer is above
    threshold. If any of the microphones are above the given level then
    recording is initiated.

    Parameters
    ----------
    mic_inputs : (Nsamples, Nchannels) np.ndarray
        Data from soundcard
    trigger_level : float
        If the input data buffer has a dB rms >= this value then True will be returned
    critical_level : float
        Critical level threshold

    Returns
    -------
    trigger_bool : bool
        True if the buffer dB rms is >= the trigger_level
    critical_bool : bool
        True if the buffer dB rms is >= the critical_level
    dBrms_channel : np.ndarray
        dB RMS values for each channel

    """
    dBrms_channel = np.apply_along_axis(calc_dBrms, 0, mic_inputs)
    trigger_bool = np.any(dBrms_channel >= trigger_level)
    critical_bool = np.any(dBrms_channel >= critical_level)
    if critical_bool:
        trigger_bool = False
    max_dBrms = np.max(dBrms_channel)

    return (trigger_bool, critical_bool, dBrms_channel)


def gcc_phat(sig, refsig, fs):
    """Compute the cross-correlation between two signals using GCC-PHAT.

    Parameters
    ----------
    sig : (Nsamples, Nchannels) np.ndarray
        First signal
    refsig : (Nsamples,) np.ndarray
        Reference signal
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    float
        Time delay in seconds between the two signals

    """
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft(sig, n=n)
    REFSIG = np.fft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.ifft(R / np.abs(R))
    max_shift = int(np.floor(n / 2))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift

    return -shift / float(fs)  # time delay


def music(
    y, fs, nch, d, bw, theta=np.linspace(-90, 90, 73), c=343, wlen=64, ns=1, show=False
):
    """Simple multiband MUltiple SIgnal Classification spatial filter implementation.

    Parameters
    ----------
    y : (Nsamples, Nchannels) np.ndarray
        Mic array signals
    fs : float
        Sampling rate in Hz
    nch : int
        Number of mics in the array
    d : float
        Mic spacing in meters
    bw : tuple
        Frequency band (low_freq, high_freq) in Hz
    theta : np.ndarray, optional
        Angle axis in degrees. Defaults to 2.5[deg] resolution
    c : float, optional
        Sound speed in m/s. Defaults to 343
    wlen : int, optional
        Window length for STFT. Defaults to 64
    ns : int, optional
        Expected number of sources. Defaults to 1
    show : bool, optional
        Plot the pseudospectrum for each band. Defaults to False

    Returns
    -------
    theta : np.ndarray
        Angle axis in degrees
    mag_p : np.ndarray
        Magnitude of spatial energy distribution estimate, averaged across bands

    """
    f_spec_axis, _, spectrum = stft(
        y, fs=fs, window=np.ones((wlen,)), nperseg=wlen, noverlap=wlen - 1, axis=0
    )
    bands = f_spec_axis[(f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1])]
    p = np.zeros_like(theta, dtype=complex)
    p_i = np.zeros((len(theta), 1), dtype=complex)

    for f_c in bands:
        w_s = 2 * np.pi * f_c * d * np.sin(np.deg2rad(theta)) / c
        a = np.exp(np.outer(np.linspace(0, nch - 1, nch), -1j * w_s))
        a_H = a.T.conj()
        spec = spectrum[f_spec_axis == f_c, :, :].squeeze()
        cov_est = np.cov(spec, bias=True)
        lambdas, V = np.linalg.eig(cov_est)
        indices = np.argsort(lambdas)[::-1]
        V_sorted = V[indices]
        V_n = V_sorted[:, ns:]
        V_n_H = V_n.T.conj()
        for i, _ in enumerate(theta):
            p_i[i] = 1 / (a_H[i, :] @ V_n @ V_n_H @ a[:, i])
            p[i] += p_i[i]
        if show:
            plt.figure()
            plt.polar(np.deg2rad(theta), 20 * np.log10(np.abs(p_i)))
            plt.xlim((-np.pi / 2, np.pi / 2))
            plt.title(str(f_c))
            plt.show()
    mag_p = np.abs(p) / len(bands)

    return theta, mag_p


def das_filter(y, fs, nch, d, bw, theta, c=343, wlen=64, show=False):
    """Simple multiband Delay and Sum spatial filter implementation.

    Parameters
    ----------
    y : (Nsamples, Nchannels) np.ndarray
        Mic array signals
    fs : float
        Sampling rate in Hz
    nch : int
        Number of mics in the array
    d : float
        Mic spacing in meters
    bw : tuple
        Frequency band (low_freq, high_freq) in Hz
    theta : np.ndarray
        Angle vector in degrees
    c : float, optional
        Sound speed in m/s. Defaults to 343
    wlen : int, optional
        Window length for STFT. Defaults to 64
    show : bool, optional
        Plot the pseudospectrum for each band. Defaults to False

    Returns
    -------
    theta : np.ndarray
        Angle axis in degrees
    mag_p : np.ndarray
        Magnitude of average spatial energy distribution estimation across bands
    f_spec_axis : np.ndarray
        Frequency axis from STFT
    spectrum : np.ndarray
        STFT spectrum
    bands : np.ndarray
        Frequency bands within the specified bandwidth

    """
    f_spec_axis, _, spectrum = stft(
        y, fs=fs, window=np.ones((wlen,)), nperseg=wlen, noverlap=wlen - 1, axis=0
    )
    bands = f_spec_axis[(f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1])]
    p = np.zeros_like(theta, dtype=complex)
    p_i = np.zeros_like(theta, dtype=complex)
    band_idxs = np.where((f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1]))[0]
    nch_range = np.linspace(
        nch - 1, 0, nch
    )  # np.linspace( nch-1,0, nch) = -90>0>90; np.linspace( 0,nch-1, nch) = 90>0>-90
    theta_rad = np.deg2rad(theta)
    w_s_mat = (
        2 * np.pi * bands[:, None] * d * np.sin(theta_rad) / c
    )  # shape: (n_bands, len(theta))
    a_mat = np.exp(np.outer(nch_range, -1j * w_s_mat.ravel())).reshape(
        nch, len(bands), len(theta)
    )  # (nch, n_bands, len(theta))
    a_H_mat = np.conj(a_mat.transpose(1, 2, 0))  # (n_bands, len(theta), nch)

    for b_idx, f_idx in enumerate(band_idxs):
        spec = spectrum[f_idx, :, :]  # (nch, n_frames)
        cov_est = np.cov(spec, bias=True)
        a = a_mat[:, b_idx, :]  # (nch, len(theta))
        a_H = a_H_mat[b_idx, :, :]  # (len(theta), nch)
        p_i = np.einsum("ij,jk,ki->i", a_H, cov_est, a) / (nch**2)
        p += p_i

        if show:
            ax.plot(
                theta_rad, 10 * np.log10(np.abs(p_i)), label=f"{bands[b_idx]:.1f} Hz"
            )
    mag_p = np.abs(p) / len(bands)

    if show:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.set_xlim((-np.pi / 2, np.pi / 2))
        plt.ylabel("Magnitude (dB)")
        ax.set_title("Pseudospectra DAS")
        ax.set_theta_offset(np.pi / 2)
        plt.legend()
        # plt.savefig("das_pseudospectra.png")
        plt.show()

    return theta, mag_p, f_spec_axis, spectrum, bands


# Other functions


def get_card(device_list):
    """Get the index of the ASIO card in the device list.

    Parameters
    ----------
    device_list : list
        List of devices (usually from sd.query_devices())

    Returns
    -------
    int or None
        Index of the MCHStreamer card in the device list, or None if not found

    """
    for i, each in enumerate(device_list):
        dev_name = each["name"]
        name = "MCHStreamer" in dev_name
        if name:
            return i
    return None

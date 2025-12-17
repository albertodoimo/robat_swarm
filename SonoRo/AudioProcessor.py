#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created: 2025-12-16
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

"""

from utilities import *
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf
import queue
import datetime
import time
import time


class AudioProcessor:
    """AudioProcessor class for real-time audio signal processing and analysis.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    channels : int
        Number of audio channels.
    block_size : int
        Size of each audio block in samples.
    analyzed_buffer_time : float
        Duration of buffer to analyze in seconds.
    data : np.ndarray
        Audio data array.
    args : object
        Command-line arguments or configuration object.
    trigger_level : float
        Sound pressure level (dB SPL) threshold to trigger detection.
    critical_level : float
        Critical sound pressure level (dB SPL) for avoiding collisions.
    mic_spacing : float
        Physical spacing between microphones in meters.
    ref : int
        Index of the reference microphone channel.
    highpass_freq : float
        High-pass filter frequency in Hz.
    lowpass_freq : float
        Low-pass filter frequency in Hz.
    theta_das : np.ndarray
        Array of angles for beamforming in radians.
    N_peaks : int
        Number of peaks to detect in the spatial response.
    soundcard_index : int
        Index of the soundcard device to use.
    interp_sensitivity : np.ndarray
        Frequency-wise sensitivity interpolation values.
    tgtmic_relevant_freqs : np.ndarray
        Indices of relevant frequency bands for target microphone.
    filename : str
        Path to the output audio file for recording.
    rec_samplerate : int
        Sample rate for recording in Hz.
    sos : np.ndarray
        Second-order sections coefficients for the IIR filter.

    Attributes
    ----------
    shared_audio_queue : queue.Queue
        Thread-safe queue for audio buffer exchange.
    current_frame : int
        Current frame position in playback.
    buffer : np.ndarray
        Current audio buffer with shape (block_size, channels).

    """

    def __init__(
        self,
        fs,
        channels,
        block_size,
        analyzed_buffer_time,
        data,
        args,
        trigger_level,
        critical_level,
        mic_spacing,
        ref,
        highpass_freq,
        lowpass_freq,
        theta_das,
        N_peaks,
        soundcard_index,
        subtype,
        interp_sensitivity,
        tgtmic_relevant_freqs,
        filename,
        rec_samplerate,
        sos,
    ):
        self.fs = fs
        self.channels = channels
        self.block_size = block_size
        self.analyzed_buffer_time = analyzed_buffer_time
        self.data = data
        self.args = args
        self.trigger_level = trigger_level
        self.critical_level = critical_level
        self.mic_spacing = mic_spacing
        self.ref = ref
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        self.theta_das = theta_das
        self.N_peaks = N_peaks
        self.soundcard_index = soundcard_index
        self.interp_sensitivity = interp_sensitivity
        self.tgtmic_relevant_freqs = tgtmic_relevant_freqs
        self.filename = filename
        self.rec_samplerate = rec_samplerate
        self.sos = sos

        self.current_frame = 0
        self.shared_audio_queue = queue.Queue()
        self.buffer = np.zeros((self.block_size, self.channels), dtype=np.float32)

    def continuos_recording(self):
        """Continuously records audio from the specified sound device and writes it to a file.

        Attributes Used
        ---------------
        self.filename : str
            Path to the output audio file where the recording will be saved.
        self.rec_samplerate : int
            Sample rate (Hz) for writing to the output file.
        self.channels : int
            Number of audio channels to record.
        self.fs : int
            Sample rate (Hz) for input stream acquisition.
        self.soundcard_index : int
            Index of the sound device to record from.
        self.callback_in : callable
            Callback function for handling input stream data.
        self.block_size : int
            Size of audio blocks processed per callback.
        self.shared_audio_queue : queue.Queue
            Thread-safe queue containing audio buffers from the input stream.
        self.buffer : ndarray
            Current audio buffer retrieved from the queue.

        """

        with sf.SoundFile(
            self.filename,
            mode="x",
            samplerate=self.rec_samplerate,
            channels=self.channels,
        ) as file:
            with sd.InputStream(
                samplerate=self.fs,
                device=self.soundcard_index,
                channels=self.channels,
                callback=self.callback_in,
                blocksize=self.block_size,
            ):
                while True:
                    self.buffer = self.shared_audio_queue.get()
                    file.write(self.buffer)

    def input_stream(self):
        """Continuously captures audio from the specified sound device.

        Attributes Used
        ---------------
        self.channels : int
            Number of audio channels to record.
        self.fs : int
            Sample rate (Hz) for input stream acquisition.
        self.soundcard_index : int
            Index of the sound device to record from.
        self.callback_in : callable
            Callback function for handling input stream data.
        self.block_size : int
            Size of audio blocks processed per callback.
        self.shared_audio_queue : queue.Queue
            Thread-safe queue containing audio buffers from the input stream.
        self.buffer : ndarray
            Current audio buffer retrieved from the queue.

        """
        with sd.InputStream(
            samplerate=self.fs,
            device=self.soundcard_index,
            channels=self.channels,
            callback=self.callback_in,
            blocksize=self.block_size,
        ) as in_stream:
            while in_stream.active:
                self.buffer = self.shared_audio_queue.get()

    def callback_out(self, outdata, frames, time, status):
        """Audio output callback for streaming audio data.

        Parameters
        ----------
        outdata : numpy.ndarray
            Output buffer where audio samples should be written. Shape is (frames, channels).
        frames : int
            Number of audio frames requested by the audio stream.
        time : CData
            Time information structure containing current playback time and hardware time.
        status : sd.CallbackFlags
            Status flags indicating any issues with the audio stream (e.g., underflow).
            If non-zero, indicates a problem occurred.

        Attributes Used
        ---------------
        self.data : numpy.ndarray
            Audio data array to be played back.
        self.current_frame : int
            Current position in the audio data array for playback.

        Raises
        ------
        sd.CallbackStop
            Raised when all audio data has been played to signal the audio stream to stop.

        """
        if status:
            print(status)
        chunksize = min(len(self.data) - self.current_frame, frames)
        outdata[:chunksize] = self.data[
            self.current_frame : self.current_frame + chunksize
        ]
        if chunksize < frames:
            outdata[chunksize:] = 0
            self.current_frame = 0  # Reset current_frame after each iteration
            raise sd.CallbackStop()
        self.current_frame += chunksize

    def callback_in(self, indata, frames, time, status):
        """
        Process incoming audio data from the audio stream.

        Parameters
        ----------
        indata : ndarray
            Audio data buffer containing the input samples for this block.
        frames : int
            Number of frames (samples) in the current audio block.
        time : CData
            Time information for the audio block (from PortAudio).
        status : CallbackFlags
            Status flags indicating any errors or special conditions
            (e.g., input overflow, buffer underflow).

        Attributes Used
        ---------------
        self.shared_audio_queue : queue.Queue
            Thread-safe queue for storing incoming audio buffers.

        Notes
        -----
        Execution time is approximately 0.00013 seconds per operation.

        """
        self.shared_audio_queue.put((indata).copy())

    def update_CC(self):
        """Calculates DOA using Cross-Correlation (CC) method.

        The processing pipeline includes:
        1. Applies a highpass filter using second-order sections to the input buffer
        2. Calculates frequency-wise RMS values for the reference channel
        3. Normalizes by mic sensitivity (from loaded calibration data)
        4. Computes total RMS across relevant frequency bands
        5. Converts to dB SPL level
        6. Calculates multi-channel time delays relative to the reference channel
        7. Estimates average angle from the calculated delays

        Attributes Used
        ---------------
        self.buffer : numpy.ndarray
            Input audio buffer containing the latest audio samples.
        self.sos : numpy.ndarray
            Second-order sections for the highpass filter.
        self.fs : int
            Sampling frequency of the audio data.
        self.ref : int
            Index of the reference microphone channel.
        self.interp_sensitivity : numpy.ndarray
            Interpolated sensitivity values for frequency normalization.
        self.tgtmic_relevant_freqs : list or numpy.ndarray
            Indices of relevant frequency bands for the target microphone.
        self.trigger_level : float
            Threshold level for triggering angle calculation.
        self.critical_level : float
            Critical threshold level for angle calculation.
        self.channels : int
            Number of audio channels.
        self.mic_spacing : float
            Spacing between microphones for delay calculation.

        Returns
        -------
        tuple
            A tuple containing:
            - avar_theta : float or None
                The average angle (in degrees) of the sound source direction. Returns None if the sound level is below
                both trigger and critical levels.
            - dB_SPL_level : float
                The sound pressure level in dB SPL calculated from the frequency-weighted
                RMS across relevant frequency bands for the reference channel.

        """
        in_buffer = self.buffer

        in_sig = signal.sosfiltfilt(self.sos, in_buffer, axis=0)

        centrefreqs, freqrms = calc_native_freqwise_rms(in_sig[:, self.ref], self.fs)

        freqwise_Parms = freqrms / self.interp_sensitivity

        total_rms_freqwise_Parms = np.sqrt(
            np.sum(freqwise_Parms[self.tgtmic_relevant_freqs] ** 2)
        )

        dB_SPL_level = pascal_to_dbspl(
            total_rms_freqwise_Parms
        )  # dB SPL level for reference channel

        ref_sig = in_sig[:, self.ref]
        delay_crossch = calc_multich_delays(in_sig, ref_sig, self.fs, self.ref)

        # calculate average angle for the array
        avar_theta = avar_angle(
            delay_crossch, self.channels, self.mic_spacing, self.ref
        )

        if dB_SPL_level > self.trigger_level or dB_SPL_level > self.critical_level:
            return np.rad2deg(avar_theta), dB_SPL_level
        else:
            avar_theta = None
            return avar_theta, dB_SPL_level

    def update_das(self):
        """Calculates DOA using Delay And Sum (DAS) method.

        The processing pipeline includes:
        1. High-pass filtering of the input buffer
        2. Max detection using Hilbert transform on the reference channel
        3. Full signal trimming around the maximum envelope peak
        4. Calculates frequency-wise RMS values for the reference channel
        5. Normalizes by mic sensitivity (from loaded calibration data)
        6. Computes total RMS across relevant frequency bands
        7. Converts to dB SPL level
        8. DAS beamforming (only in the filtered bandwidth) for spatial response and direction estimation
        9. Peak detection and angle extraction

        Attributes Used
        ---------------
        self.buffer : numpy.ndarray
            Input audio buffer containing the latest audio samples.
        self.fs : int
            Sampling frequency of the audio data.
        self.ref : int
            Index of the reference microphone channel.
        self.sos : numpy.ndarray
            Second-order sections for the highpass filter.
        self.analyzed_buffer_time : float
            Duration (in seconds) of the buffer to analyze.
        self.interp_sensitivity : numpy.ndarray
            Interpolated sensitivity values for frequency normalization.
        self.tgtmic_relevant_freqs : list or numpy.ndarray
            Indices of relevant frequency bands for the target microphone.
        self.trigger_level : float
            Threshold level for triggering angle calculation.
        self.critical_level : float
            Critical threshold level for angle calculation.
        self.channels : int
            Number of audio channels.
        self.mic_spacing : float
            Spacing between microphones for delay calculation.
        self.highpass_freq : float
            High-pass filter cutoff frequency.
        self.lowpass_freq : float
            Low-pass filter cutoff frequency.
        self.theta_das : numpy.ndarray
            Array of angles (in degrees) for DAS beamforming.
        self.N_peaks : int
            Number of peaks to detect in the spatial response.

        Returns
        -------
        tuple
            If the detected dB SPL level exceeds the trigger level or critical level:
                (float, float): A tuple containing:
                    - peak_angle (float): The angle (in degrees) of the strongest
                      detected acoustic source
                    - dB_SPL_level (float): The sound pressure level in dB SPL of the
                      detected event
            Otherwise:
                (None, float): A tuple containing:
                    - None: No direction detected
                    - dB_SPL_level (float): The sound pressure level in dB SPL
        Notes
        -----
        - The trimming window of self.analyzed_buffer_time seconds is centered on
          the maximum envelope amplitude
        - The DAS beamforming is performed only within the specified highpass and lowpass bandwidth
        - Only the top N peaks (defined by N_peaks) sorted by their height are considered for direction
          estimation

        """
        in_buffer = self.buffer

        in_sig = signal.sosfiltfilt(self.sos, in_buffer, axis=0)

        # Filter the input with its envelope on ref channel
        filtered_envelope = np.abs(signal.hilbert(in_sig[:, self.ref], axis=0))

        max_envelope_idx = np.argmax(filtered_envelope)

        # Trim all channels around the max
        trim_ms = self.analyzed_buffer_time  # ms
        trim_samples = int(self.fs * trim_ms)
        half_trim = trim_samples // 2
        trimmed_signal = np.zeros((trim_samples, in_sig.shape[1]), dtype=in_sig.dtype)

        # Ensure trimmed_signal always has exactly trim_samples rows (matching trim_ms duration)
        if max_envelope_idx - half_trim < 0:
            start_idx = 0
            end_idx = trim_samples
        elif max_envelope_idx + half_trim > in_sig.shape[0]:
            end_idx = in_sig.shape[0]
            start_idx = end_idx - trim_samples
        else:
            start_idx = max_envelope_idx - half_trim
            end_idx = start_idx + trim_samples

        trimmed_signal = in_sig[start_idx:end_idx, :]

        centrefreqs, freqrms = calc_native_freqwise_rms(
            trimmed_signal[:, self.ref], self.fs
        )
        freqwise_Parms = freqrms / self.interp_sensitivity

        total_rms_freqwise_Parms = np.sqrt(
            np.sum(freqwise_Parms[self.tgtmic_relevant_freqs] ** 2)
        )
        dB_SPL_level = pascal_to_dbspl(
            total_rms_freqwise_Parms
        )  # dB SPL level for reference channel

        theta, spatial_resp, f_spec_axis, spectrum, bands = das_filter(
            trimmed_signal,
            self.fs,
            self.channels,
            self.mic_spacing,
            [self.highpass_freq, self.lowpass_freq],
            theta=self.theta_das,
        )
        peaks, _ = signal.find_peaks(spatial_resp)  # Adjust height as needed

        peak_angles = theta[peaks]
        N = self.N_peaks  # Number of peaks to keep

        # Sort peaks by their height and keep the N largest ones
        peak_heights = spatial_resp[peaks]
        top_n_peak_indices = np.argsort(peak_heights)[
            -N:
        ]  # Indices of the N largest peaks # Indices of the N largest peaks
        top_n_peak_indices = top_n_peak_indices[::-1]
        peak_angles = theta[peaks[top_n_peak_indices]]  # Corresponding angles

        if dB_SPL_level > self.trigger_level or dB_SPL_level > self.critical_level:
            return peak_angles[0], dB_SPL_level
        else:
            peak_angles = None
            return peak_angles, dB_SPL_level

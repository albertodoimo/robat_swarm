from functions.das_v2 import das_filter
from functions.music import music
from functions.get_card import get_card 
from functions.pow_two_pad_and_window import pow_two_pad_and_window
from functions.check_if_above_level import check_if_above_level
from functions.calc_multich_delays import calc_multich_delays
from functions.avar_angle import avar_angle
from functions.bandpass import bandpass
from functions.save_data_to_csv import save_data_to_csv
from functions.utilities import pascal_to_dbspl, calc_native_freqwise_rms, interpolate_freq_response
from functions.save_data_to_xml import save_data_to_xml
from functions.matched_filter import matched_filter
from functions.detect_peaks import detect_peaks
from scipy.ndimage import gaussian_filter1d

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import scipy.signal as signal 
import sounddevice as sd
import soundfile as sf
import argparse
import queue
import datetime
import time
import random
import os
import pandas as pd 
import netifaces as ni
import time

# Stream callback function
class AudioProcessor:
    def __init__(self, fs, channels, block_size, analyzed_buffer_time, data, args, trigger_level, critical_level, mic_spacing, ref, highpass_freq, lowpass_freq, theta_das,
                 N_peaks, usb_fireface_index, subtype, interp_sensitivity, tgtmic_relevant_freqs, filename, rec_samplerate, sos, sweep):
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
        self.q = queue.Queue()
        self.qq = queue.Queue()
        self.qqq = queue.Queue()
        self.shared_audio_queue = queue.Queue()
        self.current_frame = 0
        self.usb_fireface_index = usb_fireface_index
        self.subtype = subtype
        self.interp_sensitivity = interp_sensitivity
        self.tgtmic_relevant_freqs = tgtmic_relevant_freqs
        self.filename = filename
        self.rec_samplerate = rec_samplerate
        self.sos = sos
        self.sweep = sweep
        self.buffer = np.zeros((self.block_size, self.channels), dtype=np.float32)

    def continuos_recording(self):
        with sf.SoundFile(self.filename, mode='x', samplerate=self.rec_samplerate,
                            channels=self.channels, subtype=self.subtype) as file:
            with sd.InputStream(samplerate=self.fs, device=self.usb_fireface_index,channels=self.channels, callback=self.callback_in, blocksize=self.block_size):
                    while True:
                        self.buffer = self.shared_audio_queue.get()
                        file.write(self.buffer)

    def input_stream(self):
        with sd.InputStream(samplerate=self.fs, device=self.usb_fireface_index,channels=self.channels, callback=self.callback_in, blocksize=self.block_size) as in_stream:
            while in_stream.active:
                self.buffer = self.shared_audio_queue.get()

    def callback_out(self, outdata, frames, time, status):
        if status:
            print(status)
        chunksize = min(len(self.data) - self.current_frame, frames)
        outdata[:chunksize] = self.data[self.current_frame:self.current_frame + chunksize]
        if chunksize < frames:
            outdata[chunksize:] = 0
            self.current_frame = 0  # Reset current_frame after each iteration
            raise sd.CallbackStop()
        self.current_frame += chunksize
            
    def callback_in(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block.
        approx only 0.00013 seconds in this operation"""
        self.shared_audio_queue.put((indata).copy())
        

    def update(self):
        start_time = time.time()
        in_buffer = self.buffer
        # print('in buffer shape', in_buffer.shape)
        start_time_1 = time.time()
        # print('buffer queue time seconds=', start_time_1 - start_time)

        # Plot and save the raw input buffer for the reference channel
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 4))
        # plt.plot(in_buffer[:, self.ref])
        # plt.title('Raw Input Buffer - Reference Channel')
        # plt.xlabel('Sample')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.savefig('raw_input_buffer_ref_channel.png')
        # plt.close()

        # Apply highpass filter to each channel using sosfiltfilt
        in_sig = signal.sosfiltfilt(self.sos, in_buffer, axis=0)
        # Apply matched filter to each channel separately

        # start_time_2 = time.time()

        # # Match filter the input with the output template to find similar sweeps
        # mf_signal = matched_filter(in_sig[:, self.ref], self.sweep)
        # peaks = detect_peaks(mf_signal, self.fs, prominence=0.5, distance=0.01)
        # if len(peaks) > 0:
        #     start_idx = peaks[0]
        #     end_idx = int(start_idx + self.sweep.shape[0])
        #     trimmed_input_peaks = in_sig[start_idx:end_idx, :]

        #     # plot trimmed input peaks 
        #     plt.figure(figsize=(10, 12))
        #     for ch in range(trimmed_input_peaks.shape[1]):
        #         ax = plt.subplot(trimmed_input_peaks.shape[1], 1, ch + 1, sharey=None if ch == 0 else plt.gca())
        #         plt.plot(trimmed_input_peaks[:, ch])
        #         plt.title(f'Trimmed Input Peaks - Channel {ch+1}')
        #         plt.xlabel('Sample')
        #         plt.grid(True)
        #         if ch == 0:
        #             plt.ylabel('Amplitude')
        #     plt.tight_layout()
        #     plt.savefig('trimmed_input_peaks_ref_channel.png')
        #     plt.close()

        # Filter the input with its envelope but without signal reference template
        # filtered_envelope = np.abs(signal.hilbert(in_sig[:, self.ref], axis = 0))
        # peaks = detect_peaks(in_sig[:, self.ref], self.fs, prominence=0.5, distance=0.01)

        start_time_3 = time.time()
        # print('matched filter time seconds=', start_time_3 - start_time_2)

        # Plot matched filtered signal and detected peaks for the reference channel
        # plt.figure(figsize=(10, 4))
        # plt.plot(in_sig[:, self.ref], label='Matched Filtered Envelope')
        # plt.plot(peaks, mf_signal[peaks], "rx", label='Detected Peaks')
        # plt.title('Matched Filtered Envelope and Detected Peaks - Reference Channel')
        # plt.xlabel('Sample')
        # plt.ylabel('Amplitude')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('matched_filtered_peaks_ref_channel.png')
        # plt.close()


        # Plot the filtered signal for the reference channel
        # plt.figure(figsize=(10, 4))
        # plt.plot(in_sig[:, self.ref])
        # plt.title('Filtered Signal - Reference Channel')
        # plt.xlabel('Sample')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.savefig('filtered_signal_ref_channel.png')
        # plt.close()

        # centrefreqs_list = []
        # freqrms_list = []
        # for ch in range(in_sig.shape[1]):
        #     centrefreqs_ch, freqrms_ch = calc_native_freqwise_rms(in_sig[:, ch], self.fs)
        #     centrefreqs_list.append(centrefreqs_ch)
        #     freqrms_list.append(freqrms_ch)
        # centrefreqs = np.array(centrefreqs_list).T
        # freqrms = np.array(freqrms_list).T

        centrefreqs, freqrms = calc_native_freqwise_rms(in_sig[:, self.ref], self.fs)
        
        freqwise_Parms = freqrms/self.interp_sensitivity
        
        start_time_4 = time.time()
        # print('rms freqwise time =', start_time_4 - start_time_3)
        # # Calculate and save the average noise spectrum (ANS) figure

        # plt.figure(figsize=(10, 4))
        # plt.plot(centrefreqs, freqwise_Parms)
        # plt.title('freqwise_Parms')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Amplitude (compensated)')
        # plt.tight_layout()
        # plt.savefig('filtered_spectrum.png')
        # plt.close()

        # Compute total RMS for each channel separately over the relevant frequency band
        # total_rms_freqwise_Parms = []
        # for ch in range(freqwise_Parms.shape[1]):
        #     relevant = self.tgtmic_relevant_freqs[:, ch]
        #     total_rms = np.sqrt(np.sum(freqwise_Parms[relevant, ch]**2))
        #     total_rms_freqwise_Parms.append(total_rms)
        # total_rms_freqwise_Parms = np.array(total_rms_freqwise_Parms)

        total_rms_freqwise_Parms = np.sqrt(np.sum(freqwise_Parms[self.tgtmic_relevant_freqs]**2))
        dB_SPL_level = pascal_to_dbspl(total_rms_freqwise_Parms) #dB SPL level for reference channel
        print('db SPL:',dB_SPL_level)
        
        ref_sig = in_sig[:,self.ref]
        delay_crossch= calc_multich_delays(in_sig,ref_sig,self.fs,self.ref)

        # calculate avarage angle
        avar_theta = avar_angle(delay_crossch,self.channels,self.mic_spacing,self.ref)
        
        time3 = datetime.datetime.now()

        avar_theta1 = np.array([np.rad2deg(avar_theta), time3.strftime('%H:%M:%S.%f')[:-3]]) # convert to degrees and add timestamp

        print('avarage theta',avar_theta1)
        
        end_time = time.time()
        # print('update time seconds =', end_time - start_time_1) 

        if dB_SPL_level > self.trigger_level or dB_SPL_level > self.critical_level:
            return np.rad2deg(avar_theta), dB_SPL_level
        else:
            avar_theta = None
            return avar_theta, dB_SPL_level

    def update_das(self):
        start_time = time.time()
        in_buffer = self.buffer
        # print('buffer queue time seconds=', start_time_1 - start_time)

        # Plot and save the raw input buffer for the reference channel
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 4))
        # plt.plot(in_buffer[:, self.ref])
        # plt.title('Raw Input Buffer - Reference Channel')
        # plt.xlabel('Sample')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.savefig('raw_input_buffer_ref_channel.png')
        # plt.close()

        # Apply highpass filter to each channel using sosfiltfilt
        in_sig = signal.sosfiltfilt(self.sos, in_buffer, axis=0)
        # Apply matched filter to each channel separately

        start_time_2 = time.time()

        # # Match filter the input with the output template to find similar sweeps
        # mf_signal = matched_filter(in_sig[:, self.ref], self.sweep)
        # peaks = detect_peaks(mf_signal, self.fs, prominence=0.5, distance=0.01)
        # if len(peaks) > 0:
        #     start_idx = peaks[0]
        #     end_idx = int(start_idx + self.sweep.shape[0])
        #     trimmed_input_peaks = in_sig[start_idx:end_idx, :]

        #     # plot trimmed input peaks 
        #     plt.figure(figsize=(10, 12))
        #     for ch in range(trimmed_input_peaks.shape[1]):
        #         ax = plt.subplot(trimmed_input_peaks.shape[1], 1, ch + 1, sharey=None if ch == 0 else plt.gca())
        #         plt.plot(trimmed_input_peaks[:, ch])
        #         plt.title(f'Trimmed Input Peaks - Channel {ch+1}')
        #         plt.xlabel('Sample')
        #         plt.grid(True)
        #         if ch == 0:
        #             plt.ylabel('Amplitude')
        #     plt.tight_layout()
        #     plt.savefig('trimmed_input_peaks_ref_channel.png')
        #     plt.close()

        # Filter the input with its envelope but without signal reference template
        filtered_envelope = np.abs(signal.hilbert(in_sig[:, self.ref], axis=0))
        # peaks = detect_peaks(in_sig[:, self.ref], self.fs, prominence=0.5, distance=0.01)

        max_envelope_idx = np.argmax(filtered_envelope)
        max_envelope_value = filtered_envelope[max_envelope_idx]
        # print('Max envelope value:', max_envelope_value)

        # Trim around the max
        trim_ms = self.analyzed_buffer_time # ms
        print('trim_ms', trim_ms)
        trim_samples = int(self.fs * trim_ms)
        print('trim_samples', trim_samples)
        half_trim = trim_samples // 2
        trimmed_signal = np.zeros((trim_samples, in_sig.shape[1]), dtype=in_sig.dtype)
        
        # Ensure trimmed_signal always has exactly trim_samples rows (matching trim_ms duration)
        if max_envelope_idx - half_trim < 0:
            start_idx = 0
            end_idx = trim_samples
            print('1')
        elif max_envelope_idx + half_trim > in_sig.shape[0]:
            end_idx = in_sig.shape[0]
            start_idx = end_idx - trim_samples
            print('2')
        else:
            start_idx = max_envelope_idx - half_trim
            end_idx = start_idx + trim_samples
            # trimmed_signal = in_sig[start_idx:end_idx, :]
            print('3')
        trimmed_signal = in_sig[start_idx:end_idx, :]

        # plot trimmed input peaks 
        plt.figure(figsize=(10, 12))
        for ch in range(trimmed_signal.shape[1]):
            ax = plt.subplot(trimmed_signal.shape[1], 1, ch + 1, sharey=None if ch == 0 else plt.gca())
            plt.plot(trimmed_signal[:, ch])
            plt.title(f'Trimmed Input Peaks - Channel {ch+1}')
            plt.xlabel('Sample')
            plt.grid(True)
            if ch == 0:
                plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig('trimmed_input.png')
        plt.close()


        start_time_3 = time.time()
        # print('matched filter time seconds=', start_time_3 - start_time_2)

        # Plot matched filtered signal and detected peaks for the reference channel
        # plt.figure(figsize=(10, 4))
        # plt.plot(in_sig[:, self.ref], label='Matched Filtered Envelope')
        # plt.plot(peaks, mf_signal[peaks], "rx", label='Detected Peaks')
        # plt.title('Matched Filtered Envelope and Detected Peaks - Reference Channel')
        # plt.xlabel('Sample')
        # plt.ylabel('Amplitude')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('matched_filtered_peaks_ref_channel.png')
        # plt.close()


        # Plot the filtered signal for the reference channel
        # plt.figure(figsize=(10, 4))
        # plt.plot(in_sig[:, self.ref])
        # plt.title('Filtered Signal - Reference Channel')
        # plt.xlabel('Sample')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.savefig('filtered_signal_ref_channel.png')
        # plt.close()

        # centrefreqs_list = []
        # freqrms_list = []
        # for ch in range(in_sig.shape[1]):
        #     centrefreqs_ch, freqrms_ch = calc_native_freqwise_rms(in_sig[:, ch], self.fs)
        #     centrefreqs_list.append(centrefreqs_ch)
        #     freqrms_list.append(freqrms_ch)
        # centrefreqs = np.array(centrefreqs_list).T
        # freqrms = np.array(freqrms_list).T
        
        centrefreqs, freqrms = calc_native_freqwise_rms(trimmed_signal[:, self.ref], self.fs)
        freqwise_Parms = freqrms/self.interp_sensitivity
        
        start_time_4 = time.time()
        # print('rms freqwise time =', start_time_4 - start_time_3)
        # # Calculate and save the average noise spectrum (ANS) figure

        # plt.figure(figsize=(10, 4))
        # plt.plot(centrefreqs, freqwise_Parms)
        # plt.title('freqwise_Parms')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.savefig('filtered_spectrum.png')
        # plt.close()

        # Compute total RMS for each channel separately over the relevant frequency band
        # total_rms_freqwise_Parms = []
        # for ch in range(freqwise_Parms.shape[1]):
        #     relevant = self.tgtmic_relevant_freqs[:, ch]
        #     total_rms = np.sqrt(np.sum(freqwise_Parms[relevant, ch]**2))
        #     total_rms_freqwise_Parms.append(total_rms)
        # total_rms_freqwise_Parms = np.array(total_rms_freqwise_Parms)

        total_rms_freqwise_Parms = np.sqrt(np.sum(freqwise_Parms[self.tgtmic_relevant_freqs]**2))
        dB_SPL_level = pascal_to_dbspl(total_rms_freqwise_Parms) #dB SPL level for reference channel
        print('db SPL:', dB_SPL_level)

        # print('time to calculate dB SPL =', time.time() - start_time_4)

        start_time_5 = time.time()
        theta, spatial_resp, f_spec_axis, spectrum, bands = das_filter(trimmed_signal, self.fs, self.channels, self.mic_spacing, [self.highpass_freq, self.lowpass_freq], theta=self.theta_das)

        # print('freq axis', f_spec_axis.shape, 'bands shape', bands.shape, 'spectrum shape', spectrum.shape)

        # plt.figure(figsize=(10, 4))
        # plt.plot(f_spec_axis, np.abs(spectrum[:, self.ref, :]))
        # plt.title('DAS Spectrum')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Amplitude ')
        # plt.tight_layout()
        # plt.savefig('das_spectrum.png')
        # plt.close()

        start_time_6 = time.time()
        # print('das computation =', start_time_6 - start_time_5)

        #spatial_resp = gaussian_filter1d(spatial_resp, sigma=4)
        peaks, _ = signal.find_peaks(spatial_resp)  # Adjust height as needed
        # peak_angle = theta_das[np.argmax(spatial_resp)]
        peak_angles = theta[peaks]
        N = self.N_peaks # Number of peaks to keep

        # Sort peaks by their height and keep the N largest ones
        peak_heights = spatial_resp[peaks]
        top_n_peak_indices = np.argsort(peak_heights)[-N:]  # Indices of the N largest peaks # Indices of the N largest peaks
        top_n_peak_indices = top_n_peak_indices[::-1]
        peak_angles = theta[peaks[top_n_peak_indices]]  # Corresponding angles
        print('peak angles', peak_angles, 'peak heights', peak_heights[top_n_peak_indices], '\n')

        end_time = time.time()

        # print('peak finding =', end_time - start_time_6)

        # print('update time seconds =', end_time - start_time)
        if dB_SPL_level > self.trigger_level or dB_SPL_level > self.critical_level:
            return peak_angles[0], dB_SPL_level
        else:
            peak_angles = None
            return peak_angles, dB_SPL_level

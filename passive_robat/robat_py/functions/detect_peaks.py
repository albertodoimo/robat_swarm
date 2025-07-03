import numpy as np
from scipy import signal
      
# Detect peaks in the matched filter output
def detect_peaks(filtered_output, sample_rate, prominence, distance):
    """
    Detect peaks in the matched filter output.

    Parameters:
    filtered_output (np.ndarray): The output of the matched filter.
    sample_rate (int): The sample rate of the audio signal.
    prominence (float): Minimum prominence of peaks to be detected.
    distance (float): Minimum distance between peaks in seconds.

    Returns:
    peaks: Indices of detected peaks in the filtered output.
    """

    peaks, properties = signal.find_peaks(filtered_output, prominence=prominence, distance=distance*sample_rate)
    return peaks
import numpy as np
from scipy import signal

def matched_filter(recording, chirp_template):

    """
    Apply a matched filter to the signal using the provided template.

    Parameters:
    recording (np.ndarray): The audio signal to be filtered.
    chirp_template (np.ndarray): The crp template used for filtering.

    Returns:
    np.ndarray: The filtered output, which is the envelope of the matched filter response.
    """

    filtered_output = np.roll(signal.correlate(recording, chirp_template, 'same', method='direct'), -len(chirp_template)//2)
    filtered_output *= signal.windows.tukey(filtered_output.size, 0.1)
    filtered_envelope = np.abs(signal.hilbert(filtered_output))
    return filtered_envelope

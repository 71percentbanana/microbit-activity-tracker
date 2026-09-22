"""
Common feature extraction module for accelerometer time-series data.
Ensures consistency between training (WISDM dataset) and real-time inference (micro:bit).
"""

import numpy as np

FEATURE_NAMES = [
    # X-axis features
    'x_mean', 'x_std', 'x_min', 'x_max', 'x_range', 'x_rms', 'x_median',
    # Y-axis features
    'y_mean', 'y_std', 'y_min', 'y_max', 'y_range', 'y_rms', 'y_median',
    # Z-axis features
    'z_mean', 'z_std', 'z_min', 'z_max', 'z_range', 'z_rms', 'z_median',
    # Magnitude features
    'mag_mean', 'mag_std', 'mag_min', 'mag_max', 'mag_range', 'mag_rms', 'mag_median'
]

def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a window of accelerometer readings.
    
    Parameters:
    -----------
    window : np.ndarray
        Array of shape (N, 3) where columns are [x, y, z] in m/s^2.
        
    Returns:
    --------
    features : np.ndarray
        1D array of 28 extracted features matching FEATURE_NAMES.
    """
    if len(window.shape) != 2 or window.shape[1] != 3:
        raise ValueError(f"Expected window of shape (N, 3), got {window.shape}")

    x = window[:, 0]
    y = window[:, 1]
    z = window[:, 2]
    mag = np.sqrt(x**2 + y**2 + z**2)

    def axis_stats(arr):
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        rms_val = np.sqrt(np.mean(arr**2))
        median_val = np.median(arr)
        return [mean_val, std_val, min_val, max_val, range_val, rms_val, median_val]

    feats = []
    feats.extend(axis_stats(x))
    feats.extend(axis_stats(y))
    feats.extend(axis_stats(z))
    feats.extend(axis_stats(mag))

    return np.array(feats, dtype=np.float32)

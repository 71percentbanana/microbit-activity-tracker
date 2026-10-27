"""
Common feature extraction module for accelerometer time-series data.
Ensures consistency between training (WISDM dataset) and real-time inference (micro:bit).

Each window gets 28 basic statistics (7 per axis and for the magnitude), plus 79 features
that describe the rhythm and shape of the movement: percentiles, skewness, kurtosis,
zero crossings, jerk, the frequency spectrum, autocorrelation (step period) and the
correlation between axes. The rhythm features vary less between people than the raw
values do, which is what lets the model work on people it was not trained on.
"""

import numpy as np

SAMPLING_RATE_HZ = 20
SIGNALS = ['x', 'y', 'z', 'mag']
BASIC_STATS = ['mean', 'std', 'min', 'max', 'range', 'rms', 'median']
# Spectral energy bands (Hz): posture/slow sway, walking, jogging, fast vibration
BANDS = [(0.3, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 10.1)]
SHAPE_STATS = (['mad', 'p10', 'p25', 'p75', 'p90', 'iqr', 'skew', 'kurtosis', 'zcr', 'jerk_std',
                'dom_freq', 'log_energy', 'spec_entropy']
               + [f'band_{lo:g}_{hi:g}hz' for lo, hi in BANDS]
               + ['acf_peak', 'step_period'])

FEATURE_NAMES = (
    [f'{s}_{stat}' for s in SIGNALS for stat in BASIC_STATS]
    + [f'{s}_{stat}' for s in SIGNALS for stat in SHAPE_STATS]
    + ['corr_xy', 'corr_xz', 'corr_yz']
)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _shape_stats(arr: np.ndarray) -> list:
    centred = arr - arr.mean()
    std = arr.std()
    p10, p25, p75, p90 = np.percentile(arr, [10, 25, 75, 90])
    skew = np.mean(centred ** 3) / std ** 3 if std > 0 else 0.0
    kurtosis = np.mean(centred ** 4) / std ** 4 - 3 if std > 0 else 0.0
    zcr = np.mean(np.diff(np.sign(centred)) != 0)
    jerk_std = np.std(np.diff(arr))

    # Frequency content: dominant frequency, total energy, how spread out it is, and per band
    spectrum = np.abs(np.fft.rfft(centred)) ** 2
    freqs = np.fft.rfftfreq(len(arr), 1 / SAMPLING_RATE_HZ)
    total = spectrum.sum() + 1e-9
    share = spectrum / total
    dom_freq = freqs[np.argmax(spectrum[1:]) + 1]
    entropy = -np.sum(share * np.log(share + 1e-12))
    bands = [spectrum[(freqs >= lo) & (freqs < hi)].sum() / total for lo, hi in BANDS]

    # Autocorrelation peak between 0.2 s and 1 s: how regular the steps are, and their period
    acf = np.correlate(centred, centred, 'full')[len(arr) - 1:]
    acf = acf / (acf[0] + 1e-9)
    lag = int(np.argmax(acf[4:20])) + 4

    return ([np.mean(np.abs(centred)), p10, p25, p75, p90, p75 - p25, skew, kurtosis, zcr, jerk_std,
             dom_freq, np.log1p(total), entropy] + bands + [acf[lag], lag / SAMPLING_RATE_HZ])


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Extract features from a window of accelerometer readings.

    Parameters:
    -----------
    window : np.ndarray
        Array of shape (N, 3) where columns are [x, y, z] in m/s^2, sampled at 20 Hz.

    Returns:
    --------
    features : np.ndarray
        1D array of 107 extracted features matching FEATURE_NAMES.
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
    for arr in (x, y, z, mag):
        feats.extend(axis_stats(arr))
    for arr in (x, y, z, mag):
        feats.extend(_shape_stats(arr.astype(np.float64)))
    feats.extend([_correlation(x, y), _correlation(x, z), _correlation(y, z)])

    return np.nan_to_num(np.array(feats, dtype=np.float32))

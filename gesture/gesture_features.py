"""
Feature extraction module for micro:bit gesture recordings.
Each recording is one gesture: an (N, 3) array of X, Y, Z accelerometer readings in milli-g,
captured while button A is held. Recordings vary in length, so every feature is either a
summary statistic or taken from a time-normalised (resampled) copy of the signal.
Ensures consistency between training (train.py) and live prediction (server.py).
"""

import numpy as np

DIGITS = [f'd{n}' for n in range(1, 10)]

# Gesture classes in display order
GESTURES = [
    'left', 'right', 'up', 'down', 'push', 'pull', 'circle',
    'shake', 'tilt_left', 'tilt_right', 'tap', 'double_tap',
] + DIGITS
GESTURE_NAMES = {
    'left': 'Swipe left',
    'right': 'Swipe right',
    'up': 'Up',
    'down': 'Down',
    'push': 'Push',
    'pull': 'Pull',
    'circle': 'Circle',
    'shake': 'Shake',
    'tilt_left': 'Tilt left',
    'tilt_right': 'Tilt right',
    'tap': 'Tap',
    'double_tap': 'Double tap',
    **{d: f'Number {d[1]}' for d in DIGITS},
}
# Short labels for confusion-matrix headers
GESTURE_SHORT = {
    'left': 'Left', 'right': 'Right', 'up': 'Up', 'down': 'Down', 'push': 'Push', 'pull': 'Pull',
    'circle': 'Circle', 'shake': 'Shake', 'tilt_left': 'Tilt L', 'tilt_right': 'Tilt R',
    'tap': 'Tap', 'double_tap': '2× Tap',
    **{d: d[1] for d in DIGITS},
}
# Dashboard grouping
GESTURE_GROUPS = {
    **{g: 'Movements' for g in ['left', 'right', 'up', 'down', 'push', 'pull', 'circle']},
    **{g: 'Shakes, tilts & taps' for g in ['shake', 'tilt_left', 'tilt_right', 'tap', 'double_tap']},
    **{d: 'Numbers (write in the air)' for d in DIGITS},
}
# Reply sent to the micro:bit so it can show the gesture on its LEDs
LED_CODES = {g: (g[1] if g in DIGITS else g.upper()) for g in GESTURES}

# Recordings shorter than this are treated as accidental button presses (~0.25 s at ~23 readings/s)
MIN_SAMPLES = 6

# Number of evenly spaced points each axis is resampled to, capturing the gesture's shape
# (16 keeps enough detail to tell air-written digits apart)
RESAMPLE_POINTS = 16

_AXES = ['x', 'y', 'z', 'mag']
_STATS = ['mean', 'std', 'min', 'max', 'range', 'argmin', 'argmax']

FEATURE_NAMES = (
    ['n_samples']
    # Summary statistics per axis (argmin/argmax are positions in time, 0 = start, 1 = end)
    + [f'{axis}_{stat}' for axis in _AXES for stat in _STATS]
    # Zero-crossing rate around the mean (separates shake from single sweeps)
    + [f'{axis}_zcr' for axis in ['x', 'y', 'z']]
    # Resampled shape: the order of peaks distinguishes left from right, up from down
    + [f'{axis}_t{i}' for axis in _AXES for i in range(RESAMPLE_POINTS)]
)


def extract_features(recording: np.ndarray) -> np.ndarray:
    """
    Extract features from one gesture recording.

    Parameters:
    -----------
    recording : np.ndarray
        Array of shape (N, 3) where columns are [x, y, z] in milli-g, N >= MIN_SAMPLES.

    Returns:
    --------
    features : np.ndarray
        1D array of len(FEATURE_NAMES) features.
    """
    recording = np.asarray(recording, dtype=np.float64)
    if recording.ndim != 2 or recording.shape[1] != 3:
        raise ValueError(f"Expected recording of shape (N, 3), got {recording.shape}")
    n = len(recording)
    if n < MIN_SAMPLES:
        raise ValueError(f"Recording has {n} samples; at least {MIN_SAMPLES} are required")

    mag = np.sqrt(np.sum(recording ** 2, axis=1))
    signals = [recording[:, 0], recording[:, 1], recording[:, 2], mag]

    def axis_stats(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        return [np.mean(arr), np.std(arr), min_val, max_val, max_val - min_val,
                np.argmin(arr) / (n - 1), np.argmax(arr) / (n - 1)]

    def zero_crossing_rate(arr):
        centred = arr - np.mean(arr)
        return np.count_nonzero(np.diff(np.sign(centred)) != 0) / (n - 1)

    def resample(arr):
        return np.interp(np.linspace(0, n - 1, RESAMPLE_POINTS), np.arange(n), arr)

    feats = [n]
    for arr in signals:
        feats.extend(axis_stats(arr))
    for arr in signals[:3]:
        feats.append(zero_crossing_rate(arr))
    for arr in signals:
        feats.extend(resample(arr))

    return np.array(feats, dtype=np.float32)

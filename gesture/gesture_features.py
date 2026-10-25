"""
Feature extraction module for micro:bit gesture recordings.
Each recording is one gesture: an (N, 3) array of X, Y, Z accelerometer readings in milli-g,
captured while button A is held. Recordings vary in length, so every feature is either a
summary statistic or taken from a time-normalised (resampled) copy of the signal.
Ensures consistency between training (train.py) and live prediction (server.py).
"""

import os
import re
import json
import numpy as np

DIGITS = [f'd{n}' for n in range(1, 10)]

# Built-in gesture classes in display order
BUILTIN_GESTURES = [
    'left', 'right', 'up', 'down', 'push', 'pull', 'circle',
    'shake', 'tilt_left', 'tilt_right', 'tap', 'double_tap',
] + DIGITS
# All gesture classes: built-in plus any custom ones added from the dashboard (see below)
GESTURES = list(BUILTIN_GESTURES)
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

# ---- Custom gestures ----
# Added from the dashboard and stored in custom_gestures.json (committed, so teammates share them).
# Their ids start with "c_"; recordings go in data/raw/<person>/<id>/ like any other gesture.
CUSTOM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_gestures.json")
CUSTOM_GROUP = 'Custom'
MAX_NAME_LENGTH = 24


def load_custom_gestures() -> list:
    try:
        with open(CUSTOM_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _apply_custom(custom: list):
    """Rebuild the gesture tables in place, so every module that imported them sees the change."""
    GESTURES[:] = BUILTIN_GESTURES + [c['id'] for c in custom]
    for table in (GESTURE_NAMES, GESTURE_SHORT, GESTURE_GROUPS, LED_CODES):
        for key in [k for k in table if k.startswith('c_')]:
            del table[key]
    for c in custom:
        GESTURE_NAMES[c['id']] = c['name']
        GESTURE_SHORT[c['id']] = c['name'][:7]
        GESTURE_GROUPS[c['id']] = CUSTOM_GROUP
        LED_CODES[c['id']] = 'CUSTOM'  # the micro:bit shows a star


def add_custom_gesture(name: str) -> str:
    """Add a custom gesture and return its id. Raises ValueError for an empty or duplicate name."""
    name = ' '.join(str(name or '').split())[:MAX_NAME_LENGTH]
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')[:20]
    if not slug:
        raise ValueError("Give the gesture a name using letters or numbers.")
    gid = 'c_' + slug
    if gid in GESTURES or name.lower() in (n.lower() for n in GESTURE_NAMES.values()):
        raise ValueError(f'A gesture called "{name}" already exists.')
    custom = load_custom_gestures() + [{'id': gid, 'name': name}]
    with open(CUSTOM_PATH, 'w', encoding='utf-8') as f:
        json.dump(custom, f, indent=2)
    _apply_custom(custom)
    return gid


def remove_custom_gesture(gid: str):
    """Remove a custom gesture from the list. Its recordings stay on disk and come back if it is re-added."""
    custom = [c for c in load_custom_gestures() if c['id'] != gid]
    with open(CUSTOM_PATH, 'w', encoding='utf-8') as f:
        json.dump(custom, f, indent=2)
    _apply_custom(custom)


_apply_custom(load_custom_gestures())

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

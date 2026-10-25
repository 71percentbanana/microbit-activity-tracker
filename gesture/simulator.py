"""
Synthetic gesture generator for demo mode (python server.py --demo).
Produces micro:bit-like accelerometer recordings (milli-g, ~23 readings/s, board held flat face up)
so the dashboard can be tried without hardware. Demo data is kept in data/sim/ and is
never mixed with real recordings in data/raw/.
"""

import os
import numpy as np

from gesture_features import GESTURES, DIGITS

# Board lying flat, face up: gravity reads about -1000 mg on Z
GRAVITY = np.array([0.0, 0.0, -1000.0])
NOISE_MG = 40
MAX_MG = 2047
SAMPLE_PERIOD = 0.043  # seconds between readings, matching the real micro:bit stream

# Air-written digits are too intricate to fake convincingly, so demo mode leaves them out
SIMULATED = [g for g in GESTURES if g not in DIGITS]


def idle_sample(rng: np.random.Generator) -> np.ndarray:
    """One reading of the board held still."""
    return np.clip(GRAVITY + rng.normal(0, NOISE_MG, 3), -MAX_MG, MAX_MG).round()


def generate(gesture: str, rng: np.random.Generator) -> np.ndarray:
    """Generate one (N, 3) recording of the given gesture."""
    if gesture in ('shake', 'circle'):
        n = int(rng.integers(25, 40))
    elif gesture in ('tap', 'tilt_left', 'tilt_right'):
        n = int(rng.integers(8, 16))
    else:
        n = int(rng.integers(10, 24))
    t = np.linspace(0, 1, n)
    sweep = np.sin(2 * np.pi * t)  # accelerate, then decelerate
    rec = np.tile(GRAVITY, (n, 1))
    # Hand tilt varies between recordings
    rec[:, 0] += rng.normal(0, 120)
    rec[:, 1] += rng.normal(0, 120)

    amp = rng.uniform(700, 1400)
    if gesture == 'left':
        rec[:, 0] -= amp * sweep
    elif gesture == 'right':
        rec[:, 0] += amp * sweep
    elif gesture == 'up':
        rec[:, 2] -= amp * sweep
    elif gesture == 'down':
        rec[:, 2] += amp * sweep
    elif gesture == 'push':
        rec[:, 1] += amp * sweep
    elif gesture == 'pull':
        rec[:, 1] -= amp * sweep
    elif gesture in ('tilt_left', 'tilt_right'):
        # Gravity rotates from Z into X as the board tips and stays there
        angle = np.clip(t * 2, 0, 1) * np.radians(rng.uniform(40, 70)) * (-1 if gesture == 'tilt_left' else 1)
        rec[:, 0] += 1000 * np.sin(angle)
        rec[:, 2] += 1000 * (1 - np.cos(angle))
    elif gesture in ('tap', 'double_tap'):
        # Short approach movement, then a sharp impact spike for each tap
        taps = [0.5] if gesture == 'tap' else [0.3, 0.75]
        for centre in taps:
            rec[:, 2] += rng.uniform(250, 450) * np.sin(2 * np.pi * np.clip((t - centre + 0.2) / 0.2, 0, 1))
            hit = int(round(centre * (n - 1)))
            rec[hit, 2] += rng.uniform(1200, 2000)
    elif gesture == 'shake':
        cycles = rng.uniform(4, 6)
        rec[:, 0] += rng.uniform(1200, 1900) * np.sin(2 * np.pi * cycles * t)
    elif gesture == 'circle':
        phase = rng.normal(0, 0.3)
        amp = rng.uniform(500, 900)
        rec[:, 0] += amp * np.sin(2 * np.pi * t + phase)
        rec[:, 1] += amp * (np.cos(2 * np.pi * t + phase) - np.cos(phase))
    else:
        raise ValueError(f"Unknown gesture: {gesture}")

    rec += rng.normal(0, NOISE_MG, rec.shape)
    return np.clip(rec, -MAX_MG, MAX_MG).round()


def seed_dataset(data_dir: str, per_gesture: int = 30, seed: int = 0):
    """Write `per_gesture` synthetic recordings of every gesture to data_dir/sim/<gesture>/."""
    rng = np.random.default_rng(seed)
    for gesture in SIMULATED:
        gesture_dir = os.path.join(data_dir, "sim", gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        for i in range(per_gesture):
            path = os.path.join(gesture_dir, f"{gesture}_sim{i:03d}.csv")
            np.savetxt(path, generate(gesture, rng), fmt='%d', delimiter=',', header='x,y,z', comments='')

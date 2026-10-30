"""
Live activity recognition for the dashboard, using the activity tracker's trained
Random Forest (../models/random_forest_har.joblib, trained on the WISDM dataset).

The micro:bit streams readings continuously, so every 0.5 s the last 10 s of readings are
resampled to the model's 20 Hz / 200-sample window, converted from milli-g to m/s², and
classified with the activity tracker's own feature extraction (../features.py).
"""

import os
import sys
import json
import time
import threading
from collections import deque, Counter
import numpy as np
import joblib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from features import extract_window_features, FEATURE_NAMES, SAMPLING_RATE_HZ, WINDOW_SAMPLES  # activity tracker features

MODEL_DIR = os.path.join(ROOT_DIR, "models")
FEATURES_PATH = os.path.join(ROOT_DIR, "data", "features.csv")
EVAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activity_eval.json")

# Display order for the activities the model knows
ACTIVITY_ORDER = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Standing', 'Sitting']

# Conversion constant: micro:bit milli-g to m/s² (1000 mg = 1g = 9.80665 m/s²)
MG_TO_MS2 = 9.80665 / 1000.0
WINDOW_S = WINDOW_SAMPLES / SAMPLING_RATE_HZ  # model window: 200 samples @ 20 Hz = 10 s
WINDOW_POINTS = WINDOW_SAMPLES
STEP_S = 0.5        # classify twice a second
MAX_GAP_S = 0.5     # skip windows where the stream paused
SMOOTHING = 3       # the displayed activity is the most common of the last 3 predictions
HISTORY_S = 600     # keep 10 minutes of predictions for the timeline


class ActivityTracker:
    """Classifies the live accelerometer stream into activities."""

    def __init__(self, broadcast):
        self.broadcast = broadcast
        self.model = None
        self.classes = []
        self.error = None
        model_path = os.path.join(MODEL_DIR, "random_forest_har.joblib")
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            self.model = joblib.load(model_path)
            self.model.n_jobs = 1  # single predictions are faster without worker threads
            self.classes = list(joblib.load(encoder_path).classes_)
        else:
            self.error = "Activity model not found - run prepare_data.py and train_model.py in the project root."
        self.samples = deque()  # (t, x, y, z) in milli-g
        self.last_eval = 0.0
        self.recent = deque(maxlen=SMOOTHING)
        self.history = deque()
        self.current = None
        self.lock = threading.Lock()

    def on_sample(self, t: float, x: float, y: float, z: float):
        """Returns the smoothed activity when it changes, else None."""
        if self.model is None:
            return None
        with self.lock:
            self.samples.append((t, x, y, z))
            while self.samples and self.samples[0][0] < t - WINDOW_S - 1.0:
                self.samples.popleft()
            if t - self.last_eval < STEP_S or t - self.samples[0][0] < WINDOW_S:
                return None
            self.last_eval = t
            window = np.array([s for s in self.samples if s[0] >= t - WINDOW_S - 0.1])
        if len(window) < 10 or np.max(np.diff(window[:, 0])) > MAX_GAP_S:
            return None

        # Resample to the model's 20 Hz grid and convert to m/s²
        grid = np.linspace(t - WINDOW_S, t, WINDOW_POINTS)
        resampled = np.column_stack([np.interp(grid, window[:, 0], window[:, k]) for k in (1, 2, 3)]) * MG_TO_MS2
        probs = self.model.predict_proba([extract_window_features(resampled.astype(np.float32))])[0]
        best = int(np.argmax(probs))
        activity = self.classes[best]

        self.recent.append(activity)
        stable = Counter(self.recent).most_common(1)[0][0]
        entry = {'t': t, 'activity': activity, 'stable': stable, 'confidence': float(probs[best])}
        self.history.append(entry)
        while self.history and self.history[0]['t'] < t - HISTORY_S:
            self.history.popleft()
        self.broadcast('activity', {**entry, 'probs': {c: float(p) for c, p in zip(self.classes, probs)}})

        changed = stable != self.current
        self.current = stable
        return stable if changed else None

    def state(self) -> dict:
        return {
            'available': self.model is not None,
            'error': self.error,
            'activities': [a for a in ACTIVITY_ORDER if a in self.classes],
            'history': list(self.history),
            'window_s': WINDOW_S,
            'step_s': STEP_S,
        }


def evaluate() -> dict:
    """
    Evaluate the activity model the same way train_model.py did (stratified 80/20 split of
    windows, random_state=42), plus a check on people the model never saw (whole users
    held out). Cached in activity_eval.json until the model file changes.
    """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    import pandas as pd

    model_path = os.path.join(MODEL_DIR, "random_forest_har.joblib")
    model_mtime = os.path.getmtime(model_path)
    if os.path.exists(EVAL_CACHE):
        with open(EVAL_CACHE) as f:
            cached = json.load(f)
        if cached.get('model_mtime') == model_mtime:
            return cached

    rf = joblib.load(model_path)
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        meta = json.load(f)
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = encoder.transform(df['activity'].to_numpy())
    classes = list(encoder.classes_)

    # Same split as train_model.py, so this is the model's own test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = rf.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    # New-user check: retrain with 20% of the people held out entirely
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, df['user']))
    held_out = RandomForestClassifier(
        n_estimators=meta.get('n_estimators', 120), max_depth=22, min_samples_split=4, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    ).fit(X[tr], y[tr])
    user_accuracy = accuracy_score(y[te], held_out.predict(X[te]))

    order = [classes.index(a) for a in ACTIVITY_ORDER if a in classes]
    cm = confusion_matrix(y_test, y_pred)
    result = {
        'model_mtime': model_mtime,
        'classes': [classes[i] for i in order],
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'user_accuracy': float(user_accuracy),
        'n_users': int(df['user'].nunique()),
        'n_test_users': int(df['user'].iloc[te].nunique()),
        'n_train': int(len(X) - len(X_test)),
        'n_test': int(len(X_test)),
        'per_class': {
            classes[i]: {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])}
            for i in order
        },
        'confusion_matrix': cm[np.ix_(order, order)].tolist(),
        'evaluated_at': time.strftime('%Y-%m-%d %H:%M'),
    }
    with open(EVAL_CACHE, 'w') as f:
        json.dump(result, f, indent=2)
    return result

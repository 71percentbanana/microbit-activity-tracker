"""
Model training and evaluation script for the micro:bit gesture classifier.
Loads recordings from data/raw/<person>/<gesture>/*.csv, extracts features, trains a
RandomForestClassifier, and saves evaluation metrics (accuracy, precision, recall,
F1, confusion matrix) plus the model used by the dashboard.

Every recording is one gesture, so the 80/20 split is made per recording and no
test recording shares samples with a training recording. When recordings come from
two or more people, each person is also left out in turn: the model is trained on
everyone else and tested on them, which shows how well it works for a new person.
"""

import os
import json
import time
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

from gesture_features import GESTURES, GESTURE_NAMES, FEATURE_NAMES, MIN_SAMPLES, extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Fewer recordings than this per gesture cannot be split and cross-validated meaningfully
MIN_PER_GESTURE = 5


def load_recordings(data_dir: str):
    """Read every data_dir/<person>/<gesture>/*.csv recording. Returns (recordings, labels, people)."""
    recordings, labels, people = [], [], []
    persons = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))) if os.path.isdir(data_dir) else []
    for person in persons:
        for gesture in GESTURES:
            gesture_dir = os.path.join(data_dir, person, gesture)
            if not os.path.isdir(gesture_dir):
                continue
            for name in sorted(os.listdir(gesture_dir)):
                if not name.endswith('.csv'):
                    continue
                rec = np.loadtxt(os.path.join(gesture_dir, name), delimiter=',', skiprows=1, ndmin=2)
                if len(rec) >= MIN_SAMPLES:
                    recordings.append(rec)
                    labels.append(gesture)
                    people.append(person)
    return recordings, np.array(labels), np.array(people)


def cross_person(X: np.ndarray, y: np.ndarray, people: np.ndarray, n_estimators: int):
    """Leave each person out in turn: train on everyone else, test on them."""
    results = {}
    for person in sorted(set(people)):
        train_mask = people != person
        # Only test on gestures the others have recorded - the model can't know the rest
        test_mask = (people == person) & np.isin(y, np.unique(y[train_mask]))
        if test_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2:
            continue
        model = new_model(n_estimators).fit(X[train_mask], y[train_mask])
        results[person] = {
            'accuracy': float(accuracy_score(y[test_mask], model.predict(X[test_mask]))),
            'n_test': int(test_mask.sum()),
            'trained_on': int(train_mask.sum()),
        }
    return results


def new_model(n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )


def train(data_dir: str = DATA_DIR, model_dir: str = MODEL_DIR, n_estimators: int = 200,
          test_size: float = 0.2, quiet: bool = False) -> dict:
    """Train, evaluate and save the model. Returns the metrics dictionary."""
    log = (lambda *a: None) if quiet else print
    os.makedirs(model_dir, exist_ok=True)

    recordings, y, people = load_recordings(data_dir)
    counts = {g: int(np.sum(y == g)) for g in GESTURES}
    classes = [g for g in GESTURES if counts[g] > 0]
    if len(classes) < 2:
        raise ValueError("Need recordings of at least 2 gestures to train.")
    too_few = [GESTURE_NAMES[g] for g in classes if counts[g] < MIN_PER_GESTURE]
    if too_few:
        raise ValueError(f"Need at least {MIN_PER_GESTURE} recordings of each gesture; too few for: {', '.join(too_few)}.")

    X = np.array([extract_features(r) for r in recordings])
    log(f"Loaded {len(X)} recordings: " + ", ".join(f"{g}={counts[g]}" for g in classes))
    people_counts = {p: int(np.sum(people == p)) for p in sorted(set(people))}
    log("People: " + ", ".join(f"{p}={n}" for p, n in people_counts.items()))

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    rf = new_model(n_estimators).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation over all recordings gives a steadier estimate than one small test set
    folds = min(5, min(counts[g] for g in classes))
    cv_scores = cross_val_score(new_model(n_estimators), X, y,
                                cv=StratifiedKFold(folds, shuffle=True, random_state=42))

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    log(f"Test accuracy: {accuracy * 100:.1f}% ({len(y_test)} test recordings)")
    log(f"{folds}-fold cross-validation accuracy: {cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%")
    by_person = cross_person(X, y, people, n_estimators) if len(people_counts) >= 2 else {}
    for p, r in by_person.items():
        log(f"Trained without {p}, tested on {p}: {r['accuracy'] * 100:.1f}% ({r['n_test']} recordings)")

    # Confusion matrix plot for the report
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[GESTURE_NAMES[g] for g in classes])
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format='d', colorbar=False)
    plt.title(f"Gesture Classifier Confusion Matrix (Acc: {accuracy * 100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # The deployed model is refit on every recording; the metrics above come from held-out data
    final = new_model(n_estimators).fit(X, y)
    joblib.dump(final, os.path.join(model_dir, "gesture_rf.joblib"))

    importances = sorted(zip(FEATURE_NAMES, final.feature_importances_), key=lambda p: p[1], reverse=True)
    metrics = {
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "trained_at": time.strftime('%Y-%m-%d %H:%M'),
        "classes": classes,
        "counts": counts,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy),
        "cv_folds": folds,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "people": people_counts,
        "cross_person": by_person,
        "cross_person_mean": float(np.mean([r['accuracy'] for r in by_person.values()])) if by_person else None,
        "per_class": {
            g: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for g, p, r, f, s in zip(classes, precision, recall, f1, support)
        },
        "confusion_matrix": cm.tolist(),
        "top_features": [{"name": n, "importance": float(v)} for n, v in importances[:10]],
    }
    with open(os.path.join(model_dir, "gesture_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved model and metrics to {model_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the micro:bit gesture classifier.")
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help="Folder of <gesture>/*.csv recordings.")
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR, help="Directory to save model artifacts.")
    parser.add_argument('--n_estimators', type=int, default=200, help="Number of trees in random forest.")
    args = parser.parse_args()

    try:
        train(args.data_dir, args.model_dir, n_estimators=args.n_estimators)
    except ValueError as e:
        print(f"Cannot train yet: {e}")
        print(f"Record gestures with the dashboard (python dashboard/server.py) - recordings are saved to {os.path.join(args.data_dir, '<your name>')}")


if __name__ == '__main__':
    main()

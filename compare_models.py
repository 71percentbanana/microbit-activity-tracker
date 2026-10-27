"""
Compares the Random Forest against other classifiers on both tasks, using exactly the same
features and splits as the rest of the project:

  Activity  - WISDM windows (data/features.csv): the stratified 80/20 split of train_model.py,
              plus the new-participant check of dashboard/activity.py (20% of users held out).
  Gestures  - micro:bit recordings (gesture/data/raw): five-fold stratified cross-validation,
              as in gesture/train.py.

Also sweeps the on-board model sizes tried by gesture/export_microbit.py and scores each one
the way the micro:bit computes it. Writes models/model_comparison.json and a chart for the report.

Usage:
  python compare_models.py
  python compare_models.py --exclude c_b     # leave out a gesture (e.g. to match an exported model)
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from features import FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "gesture"))
import train as gesture_train
import export_microbit
from gesture_features import extract_features

FEATURES_PATH = os.path.join(BASE_DIR, "data", "features.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")


def candidates(random_forest: RandomForestClassifier) -> dict:
    """Every model gets the same features; distance- and gradient-based ones are standardised first."""
    scaled = lambda m: make_pipeline(StandardScaler(), m)
    return {
        'Random Forest': random_forest,
        'Extra Trees': ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'SVM (RBF)': scaled(SVC(C=10, gamma='scale', class_weight='balanced')),
        'k-NN (k=5)': scaled(KNeighborsClassifier(n_neighbors=5)),
        'MLP (64-64)': scaled(MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, early_stopping=True, random_state=42)),
        'Logistic Regression': scaled(LogisticRegression(max_iter=2000, class_weight='balanced')),
        'Naive Bayes': GaussianNB(),
    }


def size_kb(model) -> float:
    return len(pickle.dumps(model)) / 1024


def compare_activity(log=print) -> dict:
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    # Integer labels: MLPClassifier's early stopping cannot score string labels
    y = np.unique(df['activity'].to_numpy(), return_inverse=True)[1]
    rf =RandomForestClassifier(n_estimators=120, max_depth=22, min_samples_split=4, min_samples_leaf=2,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    # Same splits as train_model.py and dashboard/activity.py
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    u_tr, u_te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, df['user']))

    results = {}
    for name, model in candidates(rf).items():
        start = time.perf_counter()
        fitted = clone(model).fit(X_tr, y_tr)
        train_s = time.perf_counter() - start
        start = time.perf_counter()
        pred = fitted.predict(X_te)
        predict_us = (time.perf_counter() - start) / len(X_te) * 1e6
        new_users = clone(model).fit(X[u_tr], y[u_tr]).predict(X[u_te])
        results[name] = {
            'accuracy': float(accuracy_score(y_te, pred)),
            'macro_f1': float(f1_score(y_te, pred, average='macro')),
            'new_user_accuracy': float(accuracy_score(y[u_te], new_users)),
            'new_user_macro_f1': float(f1_score(y[u_te], new_users, average='macro')),
            'train_s': train_s,
            'predict_us': predict_us,
            'size_kb': size_kb(fitted),
        }
        r = results[name]
        log(f"  {name:20s} acc {r['accuracy'] * 100:5.1f}%  new users {r['new_user_accuracy'] * 100:5.1f}%  "
            f"train {train_s:6.1f}s  {r['size_kb']:9.0f} KB")
    return {'n_windows': int(len(X)), 'n_test': int(len(X_te)), 'n_users': int(df['user'].nunique()),
            'n_test_users': int(df['user'].iloc[u_te].nunique()), 'models': results}


def load_gestures(exclude):
    recordings, y, people = gesture_train.load_recordings(gesture_train.DATA_DIR)
    keep = ~np.isin(y, exclude)
    recordings = [r for r, k in zip(recordings, keep) if k]
    return recordings, y[keep], people[keep]


def compare_gestures(recordings, y, log=print) -> dict:
    X = np.array([extract_features(r) for r in recordings])
    y = np.unique(y, return_inverse=True)[1]
    rf = gesture_train.new_model(200)
    folds = StratifiedKFold(5, shuffle=True, random_state=42)  # same folds as gesture/train.py

    results = {}
    for name, model in candidates(rf).items():
        scores, f1s, fit_times, predict_times = [], [], [], []
        for tr, te in folds.split(X, y):
            start = time.perf_counter()
            fitted = clone(model).fit(X[tr], y[tr])
            fit_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            pred = fitted.predict(X[te])
            predict_times.append((time.perf_counter() - start) / len(te))
            scores.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average='macro'))
        results[name] = {
            'cv_accuracy': float(np.mean(scores)),
            'cv_std': float(np.std(scores)),
            'cv_macro_f1': float(np.mean(f1s)),
            'train_s': float(np.mean(fit_times)),
            'predict_us': float(np.mean(predict_times) * 1e6),
            'size_kb': size_kb(clone(model).fit(X, y)),
        }
        r = results[name]
        log(f"  {name:20s} cv {r['cv_accuracy'] * 100:5.1f}% ± {r['cv_std'] * 100:.1f}  "
            f"train {r['train_s']:6.2f}s  {r['size_kb']:9.0f} KB")
    return {'n_recordings': int(len(y)), 'n_classes': int(len(set(y))), 'models': results}


def onboard_sizes(recordings, y, log=print) -> list:
    """Each size export_microbit.py can choose, packed and scored as the micro:bit does it."""
    X32 = np.array([extract_features(r) for r in recordings])
    rows = []
    for trees, depth in export_microbit.CANDIDATES:
        model = export_microbit.pack_forest(export_microbit.small_forest(trees, depth).fit(X32, y))
        accuracy = export_microbit.board_accuracy(recordings, y, trees, depth)
        rows.append({'trees': trees, 'max_depth': depth,
                     'nodes': (len(model) - 2 * trees) // export_microbit.NODE_BYTES,
                     'kb': len(model) / 1024, 'fits': len(model) <= export_microbit.MAX_MODEL_BYTES,
                     'cv_accuracy': float(accuracy)})
        log(f"  {trees:3d} trees, depth {depth:2d}: {rows[-1]['kb']:6.1f} KB  "
            f"{'fits' if rows[-1]['fits'] else 'too big'}  cv {accuracy * 100:.1f}%")
    return rows


def plot(activity: dict, gestures: dict, path: str):
    """Two stacked panels (one paper column) sharing the model order: activity and gestures."""
    order = sorted(activity['models'], key=lambda m: activity['models'][m]['new_user_accuracy'])
    ypos = np.arange(len(order))
    blue, orange, ink, muted = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
    plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.edgecolor': muted,
                         'axes.labelcolor': ink, 'xtick.color': muted, 'ytick.color': ink})
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.7), gridspec_kw={'height_ratios': [1.35, 1]})

    ax = axes[0]
    h = 0.38
    split = [activity['models'][m]['accuracy'] * 100 for m in order]
    users = [activity['models'][m]['new_user_accuracy'] * 100 for m in order]
    ax.barh(ypos + h / 2 + 0.01, split, h, color=blue, label='Random 80/20 split')
    ax.barh(ypos - h / 2 - 0.01, users, h, color=orange, label='Unseen participants')
    for yp, v in zip(ypos - h / 2, users):
        ax.text(v + 0.8, yp, f"{v:.1f}", va='center', fontsize=6.5, color=muted)
    ax.set_title('(a) Activities (WISDM)', fontsize=8)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5, frameon=False,
               handlelength=1, columnspacing=1.2)

    ax = axes[1]
    cv = [gestures['models'][m]['cv_accuracy'] * 100 for m in order]
    err = [gestures['models'][m]['cv_std'] * 100 for m in order]
    ax.barh(ypos, cv, 0.6, xerr=err, color=blue, error_kw={'elinewidth': 0.8, 'ecolor': muted, 'capsize': 1.5})
    for yp, v, e in zip(ypos, cv, err):
        ax.text(v + e + 0.8, yp, f"{v:.1f}", va='center', fontsize=6.5, color=muted)
    ax.set_title('(b) Gestures (micro:bit, 5-fold CV)', fontsize=8)

    for ax in axes:
        ax.set_xlim(0, 109)
        ax.set_xticks(range(0, 101, 20))
        ax.grid(axis='x', color='#e4e3df', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_yticks(ypos)
        ax.set_yticklabels(order)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[1].set_xlabel('Accuracy (%)')
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    plt.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare classifiers on the activity and gesture tasks.")
    parser.add_argument('--exclude', nargs='*', default=[], help="Gesture ids to leave out (e.g. c_b).")
    args = parser.parse_args()

    print(f"Activity recognition (WISDM, {len(FEATURE_NAMES)} features):")
    activity = compare_activity()
    recordings, y, people = load_gestures(args.exclude)
    print(f"\nGesture recognition ({len(y)} recordings, {len(set(y))} gestures, 96 features):")
    gestures = compare_gestures(recordings, y)
    print("\nOn-board model sizes (micro:bit V2):")
    sizes = onboard_sizes(recordings, y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {'activity': activity, 'gestures': gestures, 'onboard_sizes': sizes,
              'excluded_gestures': args.exclude, 'people': sorted(set(people)),
              'compared_at': time.strftime('%Y-%m-%d %H:%M')}
    with open(os.path.join(OUTPUT_DIR, "model_comparison.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    plot(activity, gestures, os.path.join(OUTPUT_DIR, "model_comparison.png"))
    print(f"\nWrote {os.path.join(OUTPUT_DIR, 'model_comparison.json')} and model_comparison.png")


if __name__ == '__main__':
    main()

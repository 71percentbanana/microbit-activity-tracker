"""
Model training and evaluation script for Human Activity Recognition (HAR).
Trains a RandomForestClassifier on extracted statistical features, generates evaluation
metrics (Accuracy, F1-Score, Confusion Matrix), and exports the production model artifact.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from features import FEATURE_NAMES

def train_and_evaluate(features_path: str, output_dir: str, n_estimators: int = 120, test_size: float = 0.2):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading extracted features from {features_path}...")
    df = pd.read_csv(features_path)

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_raw = df['activity'].to_numpy()

    # Encode activity strings to categorical integer IDs
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)
    print(f"Target classes ({len(class_names)}): {class_names}")

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Dataset split: {len(X_train):,} training windows, {len(X_test):,} testing windows.")

    # Initialize and train Random Forest
    print(f"\nTraining RandomForestClassifier (n_estimators={n_estimators}, n_jobs=-1, class_weight='balanced')...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=22,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("Model training complete!")

    # Predictions & Evaluation on unseen test set
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"   TEST SET OVERALL ACCURACY: {accuracy * 100:.2f}%")
    print(f"==========================================\n")

    report_str = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("Classification Report:")
    print(report_str)

    # Save metrics report to text file
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report_str)

    # Generate & save Confusion Matrix plot
    print("Plotting confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format='d')
    plt.title(f"HAR Random Forest Confusion Matrix (Acc: {accuracy * 100:.1f}%)")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Feature Importance Plot
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances (Random Forest)")
    plt.bar(range(len(indices)), importances[indices], align="center", color="#2b5c8f")
    plt.xticks(range(len(indices)), [FEATURE_NAMES[i] for i in indices], rotation=90)
    plt.tight_layout()
    feat_path = os.path.join(output_dir, "feature_importances.png")
    plt.savefig(feat_path, dpi=200)
    plt.close()
    print(f"Saved feature importances plot to {feat_path}")

    # Export Model Artifacts
    model_path = os.path.join(output_dir, "random_forest_har.joblib")
    encoder_path = os.path.join(output_dir, "label_encoder.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(rf, model_path)
    joblib.dump(label_encoder, encoder_path)

    metadata = {
        "model_type": "RandomForestClassifier",
        "accuracy": float(accuracy),
        "classes": class_names,
        "feature_names": FEATURE_NAMES,
        "sampling_rate_hz": 20,
        "window_size_samples": 40,
        "window_duration_seconds": 2.0,
        "step_size_samples": 20,
        "n_estimators": n_estimators
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll artifacts saved to {output_dir}:")
    print(f"  - Model: {model_path}")
    print(f"  - Label Encoder: {encoder_path}")
    print(f"  - Metadata: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Random Forest HAR Model.")
    parser.add_argument('--features_path', type=str, default="data/features.csv", help="Extracted features CSV.")
    parser.add_argument('--output_dir', type=str, default="models", help="Directory to save model artifacts.")
    parser.add_argument('--n_estimators', type=int, default=120, help="Number of trees in random forest.")
    args = parser.parse_args()

    train_and_evaluate(args.features_path, args.output_dir, n_estimators=args.n_estimators)

if __name__ == '__main__':
    main()

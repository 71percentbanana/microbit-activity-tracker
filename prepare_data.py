"""
Data preparation script for WISDM accelerometer dataset.
Parses raw data, cleans malformed entries, segments into sliding windows,
and extracts time-domain statistical features for ML training.
"""

import os
import argparse
import numpy as np
import pandas as pd
from features import extract_window_features, FEATURE_NAMES

def load_and_clean_wisdm(file_path: str) -> pd.DataFrame:
    """
    Parse the raw WISDM text file line by line, cleaning trailing semicolons
    and skipping corrupted lines.
    """
    print(f"Loading raw dataset from: {file_path}")
    records = []
    skipped = 0

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            clean = line.strip().rstrip(';').strip()
            if not clean:
                continue
            parts = clean.split(',')
            if len(parts) >= 6:
                try:
                    user_id = int(parts[0])
                    activity = parts[1].strip()
                    # Standardize any known typos in WISDM
                    if activity == 'Uptairs':
                        activity = 'Upstairs'
                    timestamp = int(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
                    # Ensure any trailing semicolon or stray char on z is stripped
                    z_str = parts[5].replace(';', '').strip()
                    z = float(z_str)
                    records.append((user_id, activity, timestamp, x, y, z))
                except (ValueError, IndexError):
                    skipped += 1
            else:
                skipped += 1

    print(f"Parsed {len(records):,} valid records (skipped {skipped:,} malformed rows).")
    df = pd.DataFrame(records, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    # Sort to maintain temporal continuity
    df.sort_values(by=['user', 'activity', 'timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def generate_window_features(df: pd.DataFrame, window_size: int = 40, step_size: int = 20) -> pd.DataFrame:
    """
    Segment the data into sliding windows without crossing user or activity boundaries.
    Extract statistical features for each window.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned accelerometer dataframe.
    window_size : int
        Window length in samples (40 samples @ 20Hz = 2.0 seconds).
    step_size : int
        Step size between windows (20 samples = 50% overlap).
    """
    print(f"Generating sliding windows (window_size={window_size}, step_size={step_size})...")
    feature_rows = []
    
    grouped = df.groupby(['user', 'activity'], sort=False)
    total_groups = len(grouped)
    
    for idx, ((user_id, activity), group) in enumerate(grouped, 1):
        accel_data = group[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        n_samples = len(accel_data)
        
        if n_samples < window_size:
            continue
            
        for start in range(0, n_samples - window_size + 1, step_size):
            window = accel_data[start : start + window_size]
            feats = extract_window_features(window)
            feature_rows.append([user_id, activity] + list(feats))

    print(f"Extracted {len(feature_rows):,} feature windows across {total_groups} user/activity sessions.")
    columns = ['user', 'activity'] + FEATURE_NAMES
    features_df = pd.DataFrame(feature_rows, columns=columns)
    return features_df

def main():
    parser = argparse.ArgumentParser(description="Preprocess WISDM dataset for HAR ML training.")
    parser.add_argument(
        '--data_path',
        type=str,
        default=r"C:\Users\alanv\Downloads\WISDM_ar_latest\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt",
        help="Path to WISDM_ar_v1.1_raw.txt"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default="data/features.csv",
        help="Path where extracted feature CSV should be saved."
    )
    parser.add_argument('--window_size', type=int, default=40, help="Window size in samples (default 40 = 2s @ 20Hz)")
    parser.add_argument('--step_size', type=int, default=20, help="Window step size in samples (default 20 = 50% overlap)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    df_clean = load_and_clean_wisdm(args.data_path)
    features_df = generate_window_features(df_clean, window_size=args.window_size, step_size=args.step_size)

    print(f"Class distribution in windows:\n{features_df['activity'].value_counts()}")
    features_df.to_csv(args.output_path, index=False)
    print(f"Saved processed features to {args.output_path}")

if __name__ == '__main__':
    main()

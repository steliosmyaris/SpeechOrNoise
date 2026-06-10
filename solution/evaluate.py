"""
evaluate.py

Runs an trained k-NN and MLP classifiers on the test audio file,
applies post-processing, computes metrics, and saves result CSVs.
Predictions are now done in BATCHES of 10,000 frames at a time.
Prints progress.
k-NN with ball_tree.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy.signal import medfilt
from sklearn.metrics import classification_report, accuracy_score

from feature_extraction import (
    load_audio, extract_features_from_audio,
    get_frame_times, FRAME_STEP, FRAME_LENGTH, SAMPLE_RATE
)

# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\user\Documents\esfkh"

TEST_AUDIO = os.path.join(
    BASE, "test-20260608T074452Z-3-001", "test", "S01_U04.CH4.wav"
)
GROUND_TRUTH_JSON = os.path.join(
    BASE, "test-20260608T074452Z-3-001", "test", "transcriptions", "S01.json"
)
MODEL_DIR  = os.path.join(BASE, "solution", "models")
OUTPUT_DIR = os.path.join(BASE, "solution", "results")

# Median filter window in frames (must be odd).
# 51 frames × 10ms = 510ms smoothing window.
MEDIAN_FILTER_FRAMES = 51

# Batch size for prediction.
# Prints the progress so I am sure it works and I' not waiting an hour
# for no reason 
PREDICT_BATCH_SIZE = 10_000

def predict_in_batches(clf, X, clf_name):
    """
    Run clf.predict() on X in chunks of PREDICT_BATCH_SIZE.
    Prints progress so you know it's not frozen.
    """
    n = len(X)
    preds = np.empty(n, dtype=int)
    n_batches = (n + PREDICT_BATCH_SIZE - 1) // PREDICT_BATCH_SIZE

    for i in range(n_batches):
        start = i * PREDICT_BATCH_SIZE
        end   = min(start + PREDICT_BATCH_SIZE, n)
        preds[start:end] = clf.predict(X[start:end])

        # Print progress every 10 batches
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            pct = 100 * (i + 1) / n_batches
            print(f"  [{clf_name}] Batch {i+1}/{n_batches} ({pct:.1f}%) "
                  f"— frames {end}/{n}")

    return preds



def load_ground_truth(json_path, audio_duration_sec):
    """
    Parse S01.json → per-frame binary label array.
    Speech frames = 1, background frames = 0.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    num_frames = int((audio_duration_sec - FRAME_LENGTH) / FRAME_STEP) + 1
    frame_labels = np.zeros(num_frames, dtype=int)

    for entry in annotations:
        if entry.get("session_id", "S01") != "S01":
            continue
        start_sec = _ts(entry["start_time"])
        end_sec   = _ts(entry["end_time"])
        start_f   = max(0, int(start_sec / FRAME_STEP))
        end_f     = min(num_frames - 1, int(end_sec / FRAME_STEP))
        frame_labels[start_f:end_f + 1] = 1

    n_speech = np.sum(frame_labels == 1)
    n_noise  = np.sum(frame_labels == 0)
    print(f"  Ground truth: {n_speech} speech frames, {n_noise} background frames")
    print(f"  Speech ratio: {100*n_speech/num_frames:.1f}%")
    return frame_labels


def _ts(ts):
    """'HH:MM:SS.ss' → float seconds"""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def apply_postprocessing(predictions):
    """
    Median filter: for each frame, take the majority vote in a window of
    MEDIAN_FILTER_FRAMES around it, so it removes short isolated wrong predictions.
    """
    smoothed = medfilt(predictions.astype(float), kernel_size=MEDIAN_FILTER_FRAMES)
    return (smoothed >= 0.5).astype(int)


def frames_to_segments(frame_labels, frame_times, audio_filename):
    """
    Merge consecutive same-label frames into segments.
    Returns a list of dicts with Audiofile, start, end, class.
    """
    rows = []
    if len(frame_labels) == 0:
        return rows

    current_label = frame_labels[0]
    current_start = 0.0

    for i in range(1, len(frame_labels)):
        if frame_labels[i] != current_label:
            rows.append({
                "Audiofile": audio_filename,
                "start": round(current_start, 3),
                "end":   round(float(frame_times[i]), 3),
                "class": "foreground" if current_label == 1 else "background"
            })
            current_label = frame_labels[i]
            current_start = float(frame_times[i])

    rows.append({
        "Audiofile": audio_filename,
        "start": round(current_start, 3),
        "end":   round(float(frame_times[-1]), 3),
        "class": "foreground" if current_label == 1 else "background"
    })
    return rows

# The complexity is high and very confusing so i print the steps 
# each time to not be confused. 
def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load models ────────────────────────────────────────────────
    print("STEP 1: Loading trained models...")
    
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    knn    = joblib.load(os.path.join(MODEL_DIR, "knn.pkl"))
    mlp    = joblib.load(os.path.join(MODEL_DIR, "mlp.pkl"))
    print("  Models loaded.")

    # ── Step 2: Load and featurize test audio ─────────────────────────────
    print("\nSTEP 2: Loading and featurizing test audio...")
    print(f"  File: {TEST_AUDIO}")
    audio = load_audio(TEST_AUDIO)
    audio_duration = len(audio) / SAMPLE_RATE
    print(f"  Duration: {audio_duration/3600:.2f} hours ({audio_duration:.1f}s)")

    print("  Extracting features (this takes ~2-3 min for 2.5h audio)...")
    X_test = extract_features_from_audio(audio)
    del audio  # we ain't need the raw audio anymore
    print(f"  Feature matrix: {X_test.shape}")

    frame_times   = get_frame_times(len(X_test))
    X_test_scaled = scaler.transform(X_test)
    del X_test  # to free more ram

    # ── Step 3: Load ground truth ─────────────────────────────────────────
    print("\nSTEP 3: Loading ground truth...")
    gt_labels = load_ground_truth(GROUND_TRUTH_JSON, audio_duration)

    # Align lengths (rounding can cause off-by-one)
    n = min(len(X_test_scaled), len(gt_labels))
    X_test_scaled = X_test_scaled[:n]
    frame_times   = frame_times[:n]
    gt_labels     = gt_labels[:n]

    audio_filename = os.path.basename(TEST_AUDIO)

    # ── Step 4: Run each classifier ───────────────────────────────────────
    for clf_name, clf in [("knn", knn), ("mlp", mlp)]:
        print(f"CLASSIFIER: {clf_name.upper()}")
        

        # Predict in batches — prints progress every 10 batches (~100k frames)
        print(f"  Predicting {n:,} frames in batches of {PREDICT_BATCH_SIZE:,}...")
        raw_preds = predict_in_batches(clf, X_test_scaled, clf_name)

        print("\n  --- Before post-processing ---")
        print(f"  Accuracy: {accuracy_score(gt_labels, raw_preds):.4f}")
        print(classification_report(gt_labels, raw_preds,
              target_names=["background", "speech"], digits=4))

        print(f"  Applying median filter "
              f"(window={MEDIAN_FILTER_FRAMES} frames = "
              f"{MEDIAN_FILTER_FRAMES*FRAME_STEP*1000:.0f}ms)...")
        smooth_preds = apply_postprocessing(raw_preds)

        print("\n  --- After post-processing ---")
        print(f"  Accuracy: {accuracy_score(gt_labels, smooth_preds):.4f}")
        print(classification_report(gt_labels, smooth_preds,
              target_names=["background", "speech"], digits=4))

        # Convert to segments and save to aCSV
        segments = frames_to_segments(smooth_preds, frame_times, audio_filename)
        df = pd.DataFrame(segments, columns=["Audiofile", "start", "end", "class"])
        out_csv = os.path.join(OUTPUT_DIR, f"results_{clf_name}.csv")
        df.to_csv(out_csv, index=False)

        print(f"\n  Saved: {out_csv}")
        print(f"  Segments: {len(df)} total "
              f"({len(df[df['class']=='foreground'])} speech, "
              f"{len(df[df['class']=='background'])} background)")
        print("\n  First 10 segments:")
        print(df.head(10).to_string(index=False))

    print("DONE. Results in:", OUTPUT_DIR)


if __name__ == "__main__":
    evaluate()
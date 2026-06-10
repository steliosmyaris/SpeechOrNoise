"""
train.py
Trains k-NN and MLP classifiers for speech vs background detection.

k-NN uses a subsample of training data (50k frames per class max)
because k-NN prediction time scales with training set size. 
Balance to equal counts per class.
MLP uses the full (balanced) dataset since it is not affected by this issue.
"""

import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from feature_extraction import collect_training_features

# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\user\Documents\esfkh"

SPEECH_DIRS = [
    os.path.join(BASE, "train-20260608T074455Z-3-001", "train", "speech"),
    os.path.join(BASE, "train-20260608T074455Z-3-002", "train", "speech"),
]

NOISE_DIRS = [
    os.path.join(BASE, "train-20260608T074455Z-3-001", "train", "noise"),
    os.path.join(BASE, "train-20260608T074455Z-3-002", "train", "noise"),
]

MODEL_DIR = os.path.join(BASE, "solution", "models")

# HOW MANY FILES TO USE FOR FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
MAX_FILES_PER_CLASS = 80

# MAX FRAMES PER CLASS FOR k-NN
# k-NN prediction time = O(n_test * n_train). 
# The MLP uses the full balanced dataset.
# ──────────────────────────────────────────────────────────────────────────────
KNN_MAX_FRAMES_PER_CLASS = 50_000


def balance_dataset(X, y, max_per_class=None, seed=42):
    """
    Balance classes by subsampling the majority class (or both) to max_per_class.
    Returns balanced X, y.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)

    # Find the smallest class size
    min_count = min(np.sum(y == c) for c in classes)
    if max_per_class is not None:
        min_count = min(min_count, max_per_class)

    indices = []
    for c in classes:
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=min_count, replace=False)
        indices.append(chosen)

    all_idx = np.concatenate(indices)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]

# Again the steps are printed clearly because of the complexity of the algorithm.
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Step 1: Extract features ───────────────────────────────────────────
    print("STEP 1: Extracting features from training audio files...")


    X, y = collect_training_features(
        speech_dirs=SPEECH_DIRS,
        noise_dirs=NOISE_DIRS,
        max_files_per_class=MAX_FILES_PER_CLASS
    )

    # ── Step 2: Normalize features ────────────────────────────────────────
    print("\nSTEP 2: Normalizing features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print("  Scaler saved.")

    # ── Step 3: Balance dataset for MLP ───────────────────────────────────
    # The noise files are shorter → fewer noise frames → imbalanced dataset.
    # We subsample speech frames down to match noise frame count.
    print("\nSTEP 3: Balancing dataset (equal speech/noise frames)...")
    X_bal, y_bal = balance_dataset(X_scaled, y, max_per_class=None)
    n_speech = np.sum(y_bal == 1)
    n_noise  = np.sum(y_bal == 0)
    print(f"  Balanced: {n_speech} speech frames, {n_noise} noise frames")

    # ── Step 4: Train/val split for MLP ───────────────────────────────────
    print("\nSTEP 4: Train/validation split (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_bal, y_bal,
        test_size=0.2,
        random_state=42,
        stratify=y_bal
    )
    print(f"  Train frames: {len(X_train)} | Val frames: {len(X_val)}")

    # ── Step 5: Train MLP (uses full balanced dataset) ────────────────────
    # Architecture: 15 inputs → 128 neurons → 64 neurons → 2 outputs
    print("\nSTEP 5: Training MLP (128→64 hidden layers)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
        n_iter_no_change=10,
    )
    mlp.fit(X_train, y_train)

    y_pred_mlp = mlp.predict(X_val)
    print("\n  MLP Validation Results:")
    print(f"  Accuracy: {accuracy_score(y_val, y_pred_mlp):.4f}")
    print(classification_report(y_val, y_pred_mlp,
          target_names=["background", "speech"], digits=4))

    joblib.dump(mlp, os.path.join(MODEL_DIR, "mlp.pkl"))
    print("  MLP model saved.")

    # ── Step 6: Train k-NN (uses small subsample) ─────────────────────────
    # WHY SUBSAMPLE FOR k-NN?
    # k-NN does not build a compact model. At prediction time it compares
    # each test frame against every training frame. With 3M training frames
    # and 950k test frames that is ~2.8 trillion comparisons.
    # With 50k training frames (50k speech + 50k noise) it is ~95 billion
    # comparisons. Accuracy stays very high because
    # the classes are well-separated in feature space. The time reduces 
    # significantly for my cheap slow and old laptop :/
    print(f"\nSTEP 6: Preparing k-NN subset "
          f"({KNN_MAX_FRAMES_PER_CLASS} frames per class)...")
    X_knn, y_knn = balance_dataset(
        X_scaled, y,
        max_per_class=KNN_MAX_FRAMES_PER_CLASS
    )
    print(f"  k-NN training set: {len(X_knn)} frames total")

    X_knn_tr, X_knn_val, y_knn_tr, y_knn_val = train_test_split(
        X_knn, y_knn,
        test_size=0.2,
        random_state=42,
        stratify=y_knn
    )

    print("  Training k-NN (k=5)...")
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean',
        algorithm='ball_tree',  # BallTree is faster than brute force for lookup
        n_jobs=-1
    )
    knn.fit(X_knn_tr, y_knn_tr)

    y_pred_knn = knn.predict(X_knn_val)
    print("\n  k-NN Validation Results:")
    print(f"  Accuracy: {accuracy_score(y_knn_val, y_pred_knn):.4f}")
    print(classification_report(y_knn_val, y_pred_knn,
          target_names=["background", "speech"], digits=4))

    joblib.dump(knn, os.path.join(MODEL_DIR, "knn.pkl"))
    print("  k-NN model saved.")


    print("TRAINING COMPLETE. Models saved to:", MODEL_DIR)



if __name__ == "__main__":
    train()
"""
feature_extraction.py

This file handles everything related to loading audio and turning it into
numbers that a classifier can work with.

Concepts:
- Frame: a tiny chunk of audio (25ms). Audio is too long to process all at once,
  so we chop it into overlapping frames and analyze each one separately.
- MFCC: Mel-Frequency Cepstral Coefficients. The standard way to describe
  what a short piece of audio "sounds like". Mimics how the human ear works.
  We extract 13 of them per frame -> each frame = a vector of 13 numbers.
- We also add Energy and Zero Crossing Rate for extra discriminative power.
  Total features per frame = 13 MFCCs + 1 energy + 1 ZCR = 15 numbers.
"""

import os
import numpy as np
import librosa  # the main audio processing library

# CONFIGURATION 
# ──────────────────────────────────────────────
SAMPLE_RATE   = 16000   # resample everything to 16kHz (standard for speech)
FRAME_LENGTH  = 0.025   # 25ms per frame
FRAME_STEP    = 0.010   # 10ms step between frames (frames overlap)
N_MFCC        = 13      # number of MFCC coefficients to extract


def load_audio(file_path):
    """
    Load a wav file and resample it to SAMPLE_RATE (16kHz).
    Returns a 1D numpy array of audio samples.
    """
    # librosa.load returns (samples_array, sample_rate)
    # sr=SAMPLE_RATE forces resampling so all files are at the same rate
    # mono=True mixes down to single channel (some files are stereo)
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio


def extract_features_from_audio(audio):
    """
    Given a raw audio array, extract features for every frame.

    Returns a 2D numpy array of shape (num_frames, num_features).
    Each row = one frame, each column = one feature.
    """
    # Convert frame length and step from seconds to samples
    n_fft    = int(FRAME_LENGTH * SAMPLE_RATE)  # samples per frame = 400
    hop_len  = int(FRAME_STEP   * SAMPLE_RATE)  # step in samples    = 160

    # 1. MFCCs 
    # librosa gives shape (N_MFCC, num_frames), we transpose to (num_frames, N_MFCC)
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=n_fft,
        hop_length=hop_len
    ).T  # shape: (num_frames, 13)

    # 2. Energy (RMS) 
    # Root Mean Square energy per frame — how loud is this frame?
    # Speech frames tend to have higher energy than silence/background.
    rms = librosa.feature.rms(
        y=audio,
        frame_length=n_fft,
        hop_length=hop_len
    ).T  # shape: (num_frames, 1)

    # 3. Zero Crossing Rate 
    # How often does the signal cross the zero line?
    # Noisy/fricative sounds cross zero more than voiced speech.
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=n_fft,
        hop_length=hop_len
    ).T  # shape: (num_frames, 1)

    #  4. Concatenate all features
    # Stack horizontally: each frame now has 13+1+1 = 15 features
    features = np.hstack([mfccs, rms, zcr])  # shape: (num_frames, 15)

    return features


def extract_features_from_file(file_path):
    """
    Convenience function: load a file and extract features in one call.
    Returns feature matrix of shape (num_frames, 15).
    """
    audio = load_audio(file_path)
    features = extract_features_from_audio(audio)
    return features


def get_frame_times(num_frames):
    """
    For a given number of frames, return the center time (in seconds)
    of each frame. Useful for converting frame indices back to timestamps.
    """
    times = np.arange(num_frames) * FRAME_STEP + FRAME_LENGTH / 2
    return times


def collect_training_features(speech_dirs, noise_dirs, max_files_per_class=80):
    """
    Walk through the training folders, extract features from wav files,
    and return labeled feature matrices ready for training.

    Parameters:
        speech_dirs       : list of folder paths containing speech wav files
        noise_dirs        : list of folder paths containing noise wav files
        max_files_per_class: how many files to use per class (keep it manageable)

    Returns:
        X : numpy array of shape (total_frames, 15) — all features
        y : numpy array of shape (total_frames,)   — labels (1=speech, 0=noise)
    """
    X_list = []
    y_list = []

    # Collect SPEECH files
    speech_files = _gather_wav_files(speech_dirs, max_files_per_class)
    print(f"[Training] Using {len(speech_files)} speech files...")

    for i, fpath in enumerate(speech_files):
        try:
            feats = extract_features_from_file(fpath)
            X_list.append(feats)
            y_list.append(np.ones(len(feats), dtype=int))  # label 1 = speech
            if (i + 1) % 10 == 0:
                print(f"  Speech: {i+1}/{len(speech_files)} files processed")
        except Exception as e:
            print(f"  Warning: could not process {fpath}: {e}")

    # Collect NOISE files 
    noise_files = _gather_wav_files(noise_dirs, max_files_per_class)
    print(f"[Training] Using {len(noise_files)} noise files...")

    for i, fpath in enumerate(noise_files):
        try:
            feats = extract_features_from_file(fpath)
            X_list.append(feats)
            y_list.append(np.zeros(len(feats), dtype=int))  # label 0 = noise
            if (i + 1) % 10 == 0:
                print(f"  Noise: {i+1}/{len(noise_files)} files processed")
        except Exception as e:
            print(f"  Warning: could not process {fpath}: {e}")

    # Stack everything into big arrays \
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print(f"\n[Training] Total frames: {len(X)} "
          f"({np.sum(y==1)} speech, {np.sum(y==0)} noise)")

    return X, y


def _gather_wav_files(directories, max_files):
    """
    Walk a list of directories recursively and collect up to max_flies wav paths.
    """
    wav_files = []
    for d in directories:
        if not os.path.isdir(d):
            print(f"  Warning: directory not found: {d}")
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, f))
                if len(wav_files) >= max_files:
                    return wav_files
    return wav_files
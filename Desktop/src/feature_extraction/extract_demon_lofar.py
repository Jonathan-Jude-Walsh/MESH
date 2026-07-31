#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

import librosa
import scipy.signal as signal

from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = "shipsear_dataset_test.csv"

DEMON_DIR = "features_demon"

LOFAR_DIR = "features_lofar"

N_FFT = 1024

HOP_LENGTH = 512

os.makedirs(
    DEMON_DIR,
    exist_ok=True
)

os.makedirs(
    LOFAR_DIR,
    exist_ok=True
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    CSV_FILE
)

print(
    f"Loaded {len(df)} samples"
)

# ============================================================
# HELPERS
# ============================================================

def extract_demon(
    y,
    sr
):
    """
    Detection of Envelope
    Modulation on Noise
    """

    analytic = signal.hilbert(y)

    envelope = np.abs(
        analytic
    )

    envelope = envelope - np.mean(
        envelope
    )

    f, t, S = signal.spectrogram(

        envelope,

        fs=sr,

        nperseg=512,

        noverlap=256
    )

    S = np.log1p(S)

    return S.astype(
        np.float32
    )


def extract_lofar(
    y,
    sr
):
    """
    LOFAR gram
    """

    S = librosa.stft(

        y,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH
    )

    S = np.abs(S)

    S = librosa.amplitude_to_db(
        S,
        ref=np.max
    )

    return S.astype(
        np.float32
    )

# ============================================================
# EXTRACTION
# ============================================================

for _, row in tqdm(
    df.iterrows(),
    total=len(df)
):

    wav_file = row["path"]

    label = row["label"]

    segment = row["segment"]

    recording = row["recording"]

    y, sr = librosa.load(
        wav_file,
        sr=None,
        mono=True
    )

    demon = extract_demon(
        y,
        sr
    )

    lofar = extract_lofar(
        y,
        sr
    )

    demon_file = os.path.join(

        DEMON_DIR,

        f"{recording}_{segment}.npz"
    )

    lofar_file = os.path.join(

        LOFAR_DIR,

        f"{recording}_{segment}.npz"
    )

    np.savez_compressed(

        demon_file,

        demon=demon,

        label=label
    )

    np.savez_compressed(

        lofar_file,

        lofar=lofar,

        label=label
    )

print()

print(
    "DEMON extraction complete."
)

print(
    "LOFAR extraction complete."
)
#!/usr/bin/env python3

import numpy as np
import librosa
import scipy.signal as signal


def extract_features(path, sr_target=None):
    """
    Extract DSP features from a WAV file.

    Returns:
        Dictionary containing:
            Time-domain features
            Frequency-domain features
            Time-frequency features
            MFCC statistics
    """

    #
    # Load audio
    #

    x, sr = librosa.load(
        path,
        mono=True,
        sr=sr_target
    )
    
    duration = len(x) / sr

    #
    # ------------------------------
    # Time-domain features
    # ------------------------------
    #

    rms = float(
        librosa.feature.rms(y=x).mean()
    )

    zcr = float(
        librosa.feature.zero_crossing_rate(x).mean()
    )

    crest_factor = float(
        np.max(np.abs(x))
        /
        (rms + 1e-12)
    )

    #
    # ------------------------------
    # Frequency-domain features
    # ------------------------------
    #

    fft = np.abs(
        np.fft.rfft(x)
    )

    freqs, psd = signal.welch(
        x,
        sr
    )

    centroid = float(
        librosa.feature.spectral_centroid(
            y=x,
            sr=sr
        ).mean()
    )

    rolloff = float(
        librosa.feature.spectral_rolloff(
            y=x,
            sr=sr
        ).mean()
    )

    harmonic = librosa.effects.harmonic(
        x
    )

    harmonic_ratio = float(
        np.sum(np.abs(harmonic))
        /
        (
            np.sum(np.abs(x))
            + 1e-12
        )
    )

    #
    # Band energy (0–1600 Hz)
    #

    freq_axis = np.fft.rfftfreq(
        len(x),
        d=1.0 / sr
    )

    band_mask = freq_axis <= 1600

    band_energy = float(
        np.sum(
            fft[band_mask] ** 2
        )
    )

    #
    # ------------------------------
    # Time-frequency features
    # ------------------------------
    #

    stft = np.abs(
        librosa.stft(x)
    )

    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=4096,
        hop_length=1024,
        n_mels=32,
        fmin=0,
        fmax=1600

    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mfcc = librosa.feature.mfcc(
        y=x,
        sr=sr,
        n_mfcc=13,
        n_fft=4096,
        hop_length=1024,
        fmax=1600

    )

    #
    # MFCC statistics
    #

    mfcc_mean = mfcc.mean(axis=1)

    mfcc_std = mfcc.std(axis=1)

    #
    # Return features
    #

    return {

        # Metadata
        "path": path,
        "sr": sr,
        "duration": duration,

        # Time domain
        "rms": rms,
        "zcr": zcr,
        "crest_factor": crest_factor,

        # Frequency domain
        "fft": fft,
        "psd": psd,
        "centroid": centroid,
        "rolloff": rolloff,
        "harmonic_ratio": harmonic_ratio,
        "band_energy": band_energy,

        # Time-frequency
        "stft": stft,
        "mel": mel,
        "logmel": logmel,
        "mfcc": mfcc,

        # MFCC summaries
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract DSP features from a WAV file"
    )

    parser.add_argument(
        "wav_file",
        help="Input WAV file"
    )

    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Optional resample rate"
    )

    args = parser.parse_args()

    feats = extract_features(
        args.wav_file,
        sr_target=args.sr
    )

    print("Features extracted")

    for key, value in feats.items():

        if isinstance(value, np.ndarray):
            print(
                f"{key:<15} shape={value.shape}"
            )
        else:
            print(
                f"{key:<15} {value}"
            )
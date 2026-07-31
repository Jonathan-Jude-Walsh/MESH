#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from features import extract_features


def build_dataset(input_dir, output_prefix, sr_target=None):

    root = Path(input_dir)

    FEATURE_DIR = Path("features")
    FEATURE_DIR.mkdir(exist_ok=True)

    wavs = sorted(
        root.rglob("segments/*.wav")
    )

    print(f"Found {len(wavs)} segments WAV segments")

    rows = []

    for i, wav in enumerate(
        tqdm(wavs, desc="Extracting features")
    ):

        try:

            feats = extract_features(
                str(wav),
                sr_target=sr_target
            )

            #
            # Expected structure:
            #
            # Standardise/
            #   Cargo/
            #       20171104a-2/
            #           segments/
            #               seg_0000.wav
            #

            try:
                label = wav.parts[-4]
                recording = wav.parts[-3]
            except IndexError:
                label = "unknown"
                recording = "unknown"

            segment = wav.stem

            #
            # Per-segment feature file
            #

            feature_file = (
                FEATURE_DIR /
                f"{label}_{recording}_{segment}.npz"
            )

            #
            # Save tensors for DL models
            #

            np.savez_compressed(
                feature_file,
                mfcc=feats["mfcc"],
                mel=feats["mel"],
                logmel=feats["logmel"],
                label=label
            )

            #
            # FFT/PSD summary statistics
            #

            fft_mean = float(
                np.mean(feats["fft"])
            )

            fft_std = float(
                np.std(feats["fft"])
            )

            psd_mean = float(
                np.mean(feats["psd"])
            )

            psd_std = float(
                np.std(feats["psd"])
            )

            #
            # CSV row
            #

            row = {

                "id": i,

                "label": label,
                "recording": recording,
                "segment": segment,

                "path": str(wav),
                "feature_file": str(feature_file),

                "sr": feats["sr"],
                "duration": feats["duration"],

                # Time-domain
                "rms": feats["rms"],
                "zcr": feats["zcr"],
                "crest_factor": feats["crest_factor"],

                # Frequency-domain
                "centroid": feats["centroid"],
                "rolloff": feats["rolloff"],
                "harmonic_ratio": feats["harmonic_ratio"],
                "band_energy": feats["band_energy"],

                # FFT summaries
                "fft_mean": fft_mean,
                "fft_std": fft_std,

                # PSD summaries
                "psd_mean": psd_mean,
                "psd_std": psd_std,
            }

            #
            # MFCC summary features
            #

            if "mfcc_mean" in feats:
                for j, value in enumerate(
                    feats["mfcc_mean"]
                ):
                    row[
                        f"mfcc_{j+1}_mean"
                    ] = float(value)

            if "mfcc_std" in feats:
                for j, value in enumerate(
                    feats["mfcc_std"]
                ):
                    row[
                        f"mfcc_{j+1}_std"
                    ] = float(value)

            rows.append(row)
            

            #
            # Checkpoint every 100 files
            #

            if len(rows) % 100 == 0:

                pd.DataFrame(rows).to_csv(
                    f"{output_prefix}_checkpoint.csv",
                    index=False
                )

                print(
                    f"Checkpoint saved: "
                    f"{len(rows)} rows"
                )
        except Exception as e:

            print()
            print(
                f"ERROR processing: {wav}"
            )
            print(e)
            print()

            continue

    #
    # Safety check
    #

    if not rows:
        print("No features extracted.")
        return

    #
    # Save CSV
    #

    df = pd.DataFrame(rows)

    csv_path = f"{output_prefix}.csv"

    df.to_csv(
        csv_path,
        index=False
    )

    #
    # Summary
    #

    print()
    print("Dataset build complete")
    print(f"CSV      : {csv_path}")
    print(f"Features : {FEATURE_DIR}")
    print(f"Rows     : {len(df)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Build ShipsEar dataset "
            "from segmented WAV files"
        )
    )

    parser.add_argument(
        "input_dir",
        help="Root Standardise directory"
    )

    parser.add_argument(
        "output_prefix",
        help="CSV output prefix"
    )

    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Optional resample rate"
    )

    args = parser.parse_args()

    build_dataset(
        input_dir=args.input_dir,
        output_prefix=args.output_prefix,
        sr_target=args.sr
    )
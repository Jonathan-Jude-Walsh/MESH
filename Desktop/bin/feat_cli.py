# feat_cli.py
import os
import glob
import argparse
import json
from tqdm import tqdm
from features import extract_features

def process_file(path, sr_target=None, json_out=None):
    feats = extract_features(path, sr_target=sr_target)

    summary = {
        "path": feats["path"],
        "sr": feats["sr"],
        "rms": float(feats["rms"]),
        "zcr": float(feats["zcr"]),
        "crest_factor": float(feats["crest_factor"]),
        "centroid": float(feats["centroid"]),
        "rolloff": float(feats["rolloff"]),
        "harmonic_ratio": float(feats["harmonic_ratio"]),
        "band_energy": float(feats["band_energy"]),
    }

    if json_out is None:
        print(json.dumps(summary, indent=2))
    else:
        json_out.append(summary)

def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for audio feature extraction."
    )
    parser.add_argument("path", help="WAV file or directory")
    parser.add_argument("--sr", type=int, default=None,
                        help="Target sample rate (e.g. 96000)")
    parser.add_argument("--json", type=str, default=None,
                        help="Optional JSON output file for batch mode")

    args = parser.parse_args()

    if os.path.isdir(args.path):
        wavs = sorted(
            glob.glob(os.path.join(args.path, "**", "*.wav"), recursive=True)
        )
        results = []
        for f in tqdm(wavs, desc="Processing files"):
            process_file(f, sr_target=args.sr, json_out=results)

        if args.json:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved JSON: {args.json}")
    else:
        process_file(args.path, sr_target=args.sr)

if __name__ == "__main__":
    main()

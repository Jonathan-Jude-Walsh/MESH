# audio_dataset.py
import os
from pathlib import Path
import glob
import torch
from torch.utils.data import Dataset
from features import extract_features

root_dir=Path("/home/jonathan/Documents/Input Dataset/Standardise/")


class AudioFeatureDataset(Dataset):
    def __init__(self, root_dir, sr_target=None, label_fn=None):
        """
        root_dir : directory with WAV files
        sr_target : resample target (e.g. 96000) or None
        label_fn : function(path) -> label (optional)
        """
        self.files = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
        )
        self.sr_target = sr_target
        self.label_fn = label_fn

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        feats = extract_features(path, sr_target=self.sr_target)

        # Example feature vector: scalar features only
        x = torch.tensor([
            feats["rms"],
            feats["zcr"],
            feats["crest_factor"],
            feats["centroid"],
            feats["rolloff"],
            feats["harmonic_ratio"],
            feats["band_energy"],
        ], dtype=torch.float32)

        if self.label_fn is not None:
            y = self.label_fn(path)
            y = torch.tensor(y, dtype=torch.long)
            return x, y, path

        return x, path

#!/usr/bin/env python3

import os
import pandas as pd

src = pd.read_csv(
    "deep_learning_dataset.csv"
)

rows = []

for _, row in src.iterrows():

    base = os.path.basename(
        row["feature_file"]
    )

    #
    # Cargo_20171104a-2_seg_0000.npz
    # -> 20171104a-2_seg_0000.npz
    #

    base = base.split(
        "_",
        1
    )[1]

    demon_file = os.path.join(
        "features_demon",
        base
    )

    if os.path.exists(
        demon_file
    ):

        rows.append({

            "label":
                row["label"],

            "recording":
                row["recording"],

            "feature_file":
                demon_file,
        })

df = pd.DataFrame(
    rows
)

df.to_csv(
    "deep_learning_dataset_demon.csv",
    index=False
)

print(
    f"Saved {len(df)} rows"
)
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

    base = base.split(
        "_",
        1
    )[1]

    lofar_file = os.path.join(
        "features_lofar",
        base
    )

    if os.path.exists(
        lofar_file
    ):

        rows.append({

            "label":
                row["label"],

            "recording":
                row["recording"],

            "feature_file":
                lofar_file,
        })

df = pd.DataFrame(
    rows
)

df.to_csv(
    "deep_learning_dataset_lofar.csv",
    index=False
)

print(
    f"Saved {len(df)} rows"
)
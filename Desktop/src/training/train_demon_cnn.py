#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
)

from src.core.paths import (
    DEMON_DATASET_PATH
)

from src.core.settings import SETTINGS

settings = SETTINGS

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = DEMON_DATASET_PATH

DROPOUT = (
    settings["demon"]["dropout"]
)

N_SPLITS = (
    settings["validation"]["n_splits"]
)

EPOCHS = (
    settings["training"]["epochs"]
)

PATIENCE = (
    settings["training"]["patience"]
)

VALIDATION_SPLIT = (
    settings["training"]["validation_split"]
)

BATCH_SIZE = (
    settings["training"]["batch_size"]
)
RANDOM_SEED = (
    settings["validation"]["random_seed"]
)

TARGET_H = (
    settings["training"]["target_height"]
)
TARGET_W = (
    settings["training"]["target_width"]
)

# ============================================================

np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(CSV_FILE)

print(
    f"Rows: {len(df)}"
)

encoder = LabelEncoder()

y = encoder.fit_transform(
    df["label"]
)

num_classes = len(
    encoder.classes_
)

# ============================================================
# FEATURES
# ============================================================

X = []

for file in df["feature_file"]:

    data = np.load(file)

    feat = data["demon"].astype(
        np.float32
    )

    feat = (
        feat - np.mean(feat)
    ) / (
        np.std(feat) + 1e-8
    )

    feat = tf.image.resize(

        feat[..., np.newaxis],

        (
            TARGET_H,
            TARGET_W
        )

    ).numpy()

    X.append(feat)

X = np.array(
    X,
    dtype=np.float32
)

print(
    "Tensor:",
    X.shape
)

# ============================================================
# MODEL
# ============================================================

def build_model(
    input_shape,
    num_classes
):

    model = Sequential([

        Input(
            shape=input_shape
        ),

        Conv2D(
            16,
            3,
            padding="same",
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling2D(),

        Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling2D(),

        Conv2D(
            64,
            3,
            padding="same",
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling2D(),

        Flatten(),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(DROPOUT),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    return model

# ============================================================
# KFOLD
# ============================================================

groups = df["recording"]

gkf = GroupKFold(
    n_splits=N_SPLITS
)

scores = []

for fold, (
    train_idx,
    test_idx

) in enumerate(

    gkf.split(
        X,
        y,
        groups
    ),

    start=1
):

    print()
    print(
        f"Fold {fold}"
    )

    model = build_model(

        X.shape[1:],

        num_classes
    )

    early = EarlyStopping(

        monitor="val_loss",

        patience=PATIENCE,

        restore_best_weights=True
    )

    model.fit(

        X[train_idx],

        y[train_idx],

        validation_split=0.1,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early
        ],

        verbose=1
    )

    pred = model.predict(
        X[test_idx],
        verbose=0
    )

    pred = np.argmax(
        pred,
        axis=1
    )

    acc = accuracy_score(
        y[test_idx],
        pred
    )

    scores.append(acc)

    print(
        f"Fold Accuracy: {acc:.4f}"
    )

print()
print(
    "Mean Accuracy:",
    np.mean(scores)
)

print(
    "Std Accuracy:",
    np.std(scores)
)
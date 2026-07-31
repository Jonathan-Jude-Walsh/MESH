#!/usr/bin/env python3

import os
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
    CNN_RESULTS_PATH
)

from src.core.settings import SETTINGS

settings = SETTINGS


# ============================================================
# CONFIG
# ============================================================

CSV_FILE = CNN_RESULTS_PATH

N_SPLITS = (
    settings["validation"]["n_splits"]
)

EPOCHS = (
    settings["training"]["epochs"]
)

BATCH_SIZE = (
    settings["training"]["batch_size"]
)

CNN_FILTERS_1 = (
    settings["cnn"]["filters_1"]
)

CNN_FILTERS_2 = (
    settings["cnn"]["filters_2"]
)

CNN_FILTERS_3 = (
    settings["cnn"]["filters_3"]
)

CNN_DENSE_UNITS = (
    settings["cnn"]["dense_units"]
)

CNN_DROPOUT = (
    settings["cnn"]["dropout"]
)

RANDOM_SEED = (
    settings["validation"]["random_seed"]
)

FEATURE_TYPE = (
    settings["training"]["feature_type"]
)
# logmel
# mel
# mfcc

TARGET_WIDTH = (
    settings["training"]["target_width"]
)

TARGET_HEIGHT = (
    settings["training"]["target_height"]
)



# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)


# ============================================================
# LOAD CSV
# ============================================================

print("Loading dataset...")

df = pd.read_csv(CSV_FILE)

print(f"Rows: {len(df)}")

print(
    f"Unique recordings: "
    f"{df['recording'].nunique()}"
)


# ============================================================
# LABELS
# ============================================================

encoder = LabelEncoder()

y = encoder.fit_transform(
    df["label"]
)

num_classes = len(
    encoder.classes_
)

print()

print("Classes:")

for i, c in enumerate(
    encoder.classes_
):
    print(i, c)


# ============================================================
# LOAD FEATURES
# ============================================================

print()

print(
    f"Loading {FEATURE_TYPE} features..."
)

X = []

for file in df["feature_file"]:

    data = np.load(
        file,
        allow_pickle=True
    )

    feat = data[FEATURE_TYPE].astype(np.float32)

    feat = (
        feat - np.mean(feat)
    ) / (
        np.std(feat) + 1e-8
    )
    #
    # Skip malformed segments
    #


    if feat.shape[1] < TARGET_WIDTH:

        feat = np.pad(
            feat,
            (
                (0,0),
                (
                    0,
                    TARGET_WIDTH - feat.shape[1]
                )
            ),
            mode="constant"
        )

    elif feat.shape[1] > TARGET_WIDTH:

        feat = feat[:, :TARGET_WIDTH]


    X.append(feat)

X = np.stack(X)

print(
    "Final feature tensor:",
    X.shape
)

#
# CNN channel dimension
#

X = X[..., np.newaxis]

print(
    "CNN shape:",
    X.shape
)


# ============================================================
# MODEL
# ============================================================

def build_cnn(
    input_shape,
    num_classes
):

    model = Sequential(

        [

            Conv2D(
                CNN_FILTERS_1,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),

            BatchNormalization(),

            MaxPooling2D(
                (2, 2)
            ),

            Conv2D(
                CNN_FILTERS_2,
                (3, 3),
                padding="same",
                activation="relu",
            ),

            BatchNormalization(),

            MaxPooling2D(
                (2, 2)
            ),

            Conv2D(
                CNN_FILTERS_3,
                (3, 3),
                padding="same",
                activation="relu",
            ),

            BatchNormalization(),

            MaxPooling2D(
                (2, 2)
            ),

            Flatten(),

            Dense(
                CNN_DENSE_UNITS,
                activation="relu"
            ),

            Dropout(
                CNN_DROPOUT
            ),

            Dense(
                num_classes,
                activation="softmax"
            ),
        ]
    )

    model.compile(

        optimizer="adam",

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ],
    )

    return model


# ============================================================
# GROUP KFOLD
# ============================================================

groups = df["recording"]

gkf = GroupKFold(
    n_splits=N_SPLITS
)

accuracies = []
precisions = []
recalls = []
f1s = []

print()
print("=" * 60)
print("GROUP KFOLD CNN")
print("=" * 60)

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
        f"Fold {fold}/{N_SPLITS}"
    )

    X_train = X[
        train_idx
    ]

    X_test = X[
        test_idx
    ]

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]

    model = build_cnn(
        X_train.shape[1:],
        num_classes
    )

    early_stop = (
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    )

    history = model.fit(

        X_train,
        y_train,

        validation_split=0.1,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stop
        ],

        verbose=1,
    )

    pred = model.predict(
        X_test,
        verbose=0
    )

    pred = np.argmax(
        pred,
        axis=1
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    prec = precision_score(
        y_test,
        pred,
        average="weighted",
        zero_division=0
    )

    rec = recall_score(
        y_test,
        pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        pred,
        average="weighted"
    )

    accuracies.append(acc)

    precisions.append(
        prec
    )

    recalls.append(
        rec
    )

    f1s.append(
        f1
    )

    print(
        f"Fold Accuracy = "
        f"{acc:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL CNN RESULTS")
print("=" * 60)

print(
    f"Accuracy  : "
    f"{np.mean(accuracies):.4f}"
)

print(
    f"Accuracy Std : "
    f"{np.std(accuracies):.4f}"
)

print(
    f"Precision : "
    f"{np.mean(precisions):.4f}"
)

print(
    f"Recall    : "
    f"{np.mean(recalls):.4f}"
)

print(
    f"F1 Score  : "
    f"{np.mean(f1s):.4f}"
)

results = pd.DataFrame(

    [

        {

            "Model":
            "CNN",

            "Feature":
            FEATURE_TYPE,

            "Accuracy":
            np.mean(
                accuracies
            ),

            "AccuracyStd":
            np.std(
                accuracies
            ),

            "Precision":
            np.mean(
                precisions
            ),

            "Recall":
            np.mean(
                recalls
            ),

            "F1":
            np.mean(
                f1s
            ),
        }

    ]

)

results.to_csv(
    "cnn_results.csv",
    index=False
)

print()

print(
    "Saved: cnn_results.csv"
)
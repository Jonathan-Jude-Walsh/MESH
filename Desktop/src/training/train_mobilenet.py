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

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
)
from src.core.paths import DL_DATASET_PATH 
from src.core.settings import SETTINGS

settings = SETTINGS

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = DL_DATASET_PATH

RANDOM_SEED = (
    settings["validation"]["random_seed"]
)

N_SPLITS = (
    settings["validation"]["n_splits"]
)

FEATURE_TYPE = settings["training"]["feature_type"]

TARGET_WIDTH = settings["training"]["target_width"]

IMAGE_SIZE = settings["mobilenet"]["image_size"]

EPOCHS = settings["training"]["epochs"]

BATCH_SIZE = settings["training"]["batch_size"]

LEARNING_RATE = settings["training"]["learning_rate"]

MOBILENET_DROPOUT = settings["mobilenet"]["dropout"]

DENSE_UNITS = settings["mobilenet"]["dense_units"]

# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(CSV_FILE)

print(f"Rows: {len(df)}")

print(
    f"Unique recordings: "
    f"{df['recording'].nunique()}"
)

# ============================================================
# LABEL ENCODING
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

for idx, cls in enumerate(
    encoder.classes_
):
    print(
        f"{idx}: {cls}"
    )

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

    feat = data[
        FEATURE_TYPE
    ].astype(
        np.float32
    )

    #
    # Per-sample normalization
    #

    feat = (
        feat - np.mean(feat)
    ) / (
        np.std(feat) + 1e-8
    )

    #
    # Pad / crop width
    #

    if feat.shape[1] < TARGET_WIDTH:

        feat = np.pad(
            feat,
            (
                (0, 0),
                (
                    0,
                    TARGET_WIDTH - feat.shape[1]
                ),
            ),
            mode="constant",
        )

    elif feat.shape[1] > TARGET_WIDTH:

        feat = feat[
            :,
            :TARGET_WIDTH
        ]

    X.append(feat)

X = np.stack(X)

print()
print(
    "Feature Tensor:",
    X.shape
)

# ============================================================
# CONVERT TO MOBILENET FORMAT
# ============================================================

X = X[..., np.newaxis]

resized = []

print()
print(
    "Resizing to MobileNet input..."
)

for sample in X:

    sample = tf.image.resize(
        sample,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    ).numpy()

    sample = np.repeat(
        sample,
        3,
        axis=-1
    )

    resized.append(sample)

X = np.array(
    resized,
    dtype=np.float32
)

print(
    "MobileNet Shape:",
    X.shape
)

# ============================================================
# MODEL
# ============================================================

def build_mobilenet(
    input_shape,
    num_classes
):

    base_model = MobileNetV2(

        input_shape=input_shape,

        include_top=False,

        weights=None
    )

    inputs = Input(
        shape=input_shape
    )

    x = base_model(
        inputs
    )

    x = GlobalAveragePooling2D()(
        x
    )

    x = Dropout(
        MOBILENET_DROPOUT
    )(x)

    x = Dense(
        DENSE_UNITS,
        activation="relu"
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

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
print("MOBILENETV2 GROUP KFOLD")
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

    model = build_mobilenet(
        X_train.shape[1:],
        num_classes
    )

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True
    )

    model.fit(

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
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        average="weighted",
        zero_division=0
    )

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

    print(
        f"Fold Accuracy = "
        f"{acc:.4f}"
    )

# ============================================================
# FINAL RESULTS
# ============================================================

accuracy = np.mean(
    accuracies
)

accuracy_std = np.std(
    accuracies
)

precision = np.mean(
    precisions
)

recall = np.mean(
    recalls
)

f1 = np.mean(
    f1s
)

print()
print("=" * 60)
print("FINAL MOBILENET RESULTS")
print("=" * 60)

print(
    f"Accuracy     : "
    f"{accuracy:.4f}"
)

print(
    f"Accuracy Std : "
    f"{accuracy_std:.4f}"
)

print(
    f"Precision    : "
    f"{precision:.4f}"
)

print(
    f"Recall       : "
    f"{recall:.4f}"
)

print(
    f"F1 Score     : "
    f"{f1:.4f}"
)

results = pd.DataFrame(
    [
        {
            "Model": "MobileNetV2",
            "Feature": FEATURE_TYPE,
            "Accuracy": accuracy,
            "AccuracyStd": accuracy_std,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    ]
)

results.to_csv(
    "mobilenet_results.csv",
    index=False
)

print()
print(
    "Saved: mobilenet_results.csv"
)
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

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    Add,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
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

FEATURE_TYPE = settings["training"]["feature_type"]

TARGET_WIDTH = settings["training"]["target_width"]

RANDOM_SEED = (
    settings["validation"]["random_seed"]
)

N_SPLITS = (
    settings["validation"]["n_splits"]
)

EPOCHS = settings["training"]["epochs"]

BATCH_SIZE = settings["training"]["batch_size"]

DROPOUT = settings["resnet"]["dropout"]

LEARNING_RATE = settings["training"]["learning_rate"]

# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(CSV_FILE)

print(
    f"Rows: {len(df)}"
)

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

for i, cls in enumerate(
    encoder.classes_
):
    print(i, cls)

# ============================================================
# FEATURES
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
    # Width padding
    #

    if feat.shape[1] < TARGET_WIDTH:

        feat = np.pad(
            feat,
            (
                (0, 0),
                (
                    0,
                    TARGET_WIDTH - feat.shape[1]
                )
            ),
            mode="constant"
        )

    elif feat.shape[1] > TARGET_WIDTH:

        feat = feat[
            :,
            :TARGET_WIDTH
        ]

    X.append(feat)

X = np.stack(X)

print(
    "Feature Tensor:",
    X.shape
)

X = X[..., np.newaxis]

print(
    "CNN Tensor:",
    X.shape
)

# ============================================================
# RESIDUAL BLOCK
# ============================================================

def residual_block(
    x,
    filters
):

    shortcut = x

    x = Conv2D(
        filters,
        (3,3),
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = ReLU()(x)

    x = Conv2D(
        filters,
        (3,3),
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:

        shortcut = Conv2D(
            filters,
            (1,1),
            padding="same"
        )(shortcut)

        shortcut = BatchNormalization()(
            shortcut
        )

    x = Add()(
        [x, shortcut]
    )

    x = ReLU()(x)

    return x

# ============================================================
# MODEL
# ============================================================

def build_resnet(
    input_shape,
    num_classes
):

    inputs = Input(
        shape=input_shape
    )

    x = Conv2D(
        16,
        (3,3),
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = ReLU()(x)

    x = residual_block(
        x,
        16
    )

    x = MaxPooling2D(
        (2,2)
    )(x)

    x = residual_block(
        x,
        32
    )

    x = MaxPooling2D(
        (2,2)
    )(x)

    x = residual_block(
        x,
        64
    )

    x = GlobalAveragePooling2D()(
        x
    )

    x = Dense(
        64,
        activation="relu"
    )(x)

    x = Dropout(
        DROPOUT
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
print("RESNET GROUP KFOLD")
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

    model = build_resnet(
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
# RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL RESNET RESULTS")
print("=" * 60)

print(
    f"Accuracy     : {np.mean(accuracies):.4f}"
)

print(
    f"Accuracy Std : {np.std(accuracies):.4f}"
)

print(
    f"Precision    : {np.mean(precisions):.4f}"
)

print(
    f"Recall       : {np.mean(recalls):.4f}"
)

print(
    f"F1 Score     : {np.mean(f1s):.4f}"
)

results = pd.DataFrame([
    {
        "Model": "ResNet",
        "Feature": FEATURE_TYPE,
        "Accuracy": np.mean(accuracies),
        "AccuracyStd": np.std(accuracies),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1": np.mean(f1s),
    }
])

results.to_csv(
    "resnet_results.csv",
    index=False
)

print()
print(
    "Saved: resnet_results.csv"
)
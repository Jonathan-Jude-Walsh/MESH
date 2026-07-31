#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GroupKFold
)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.core.paths import (
    CLASSICAL_DATASET_PATH
)

from src.core.settings import SETTINGS

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = CLASSICAL_DATASET_PATH

settings = SETTINGS


RANDOM_STATE = (
    settings["validation"]["random_seed"]
)

TEST_SIZE = (
    settings["validation"]["test_size"]
)

SPLIT_MODE = (
    settings["validation"]["split_mode"]
)

N_SPLITS = (
    settings["validation"]["n_splits"]
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(CSV_FILE)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# PREPARE FEATURES
# ============================================================

drop_cols = [
    "id",
    "label",
    "recording",
    "segment",
    "path",
    "feature_file"
]

feature_cols = [
    c for c in df.columns
    if c not in drop_cols
]

X = df[feature_cols].copy()

X = X.fillna(0)

y_raw = df["label"]


# ============================================================
# LABEL ENCODING
# ============================================================

encoder = LabelEncoder()

y = encoder.fit_transform(y_raw)

print()
print("Classes:")

for idx, cls in enumerate(encoder.classes_):
    print(f"{idx}: {cls}")




# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

if SPLIT_MODE == "segment":

    print()
    print("Using SEGMENT-level split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

elif SPLIT_MODE == "recording":

    print()
    print("Using RECORDING-level split")

    recordings = df["recording"].unique()

    train_recordings, test_recordings = train_test_split(
        recordings,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_mask = df["recording"].isin(
        train_recordings
    )

    test_mask = df["recording"].isin(
        test_recordings
    )

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    print(
        f"Train recordings: {len(train_recordings)}"
    )

    print(
        f"Test recordings: {len(test_recordings)}"
    )

elif SPLIT_MODE == "groupkfold":

    print()
    print(
        f"Using GroupKFold "
        f"({N_SPLITS} folds)"
    )

else:

    raise ValueError(
        f"Unknown SPLIT_MODE: {SPLIT_MODE}"
    )

# ============================================================
# SCALE FEATURES
# ============================================================

if SPLIT_MODE != "groupkfold":

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

# ============================================================
# MODELS
# ============================================================

models = {}

models["SVM"] = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

models["RandomForest"] = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

models["KNN"] = KNeighborsClassifier(
    n_neighbors=5
)

if XGBOOST_AVAILABLE:

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss"
    )


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

for name, model in models.items():

    print()
    print(f"Training {name}...")

    if SPLIT_MODE != "groupkfold":

        #
        # Existing behaviour
        #

        if name in ["SVM", "KNN"]:

            model.fit(
                X_train_scaled,
                y_train
            )

            y_pred = model.predict(
                X_test_scaled
            )

        else:

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_test
            )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

    else:

        #
        # Group K-Fold
        #

        groups = df["recording"]

        gkf = GroupKFold(
            n_splits=N_SPLITS
        )

        accuracies = []
        precisions = []
        recalls = []
        f1s = []


        
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

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train
            )

            X_test_scaled = scaler.transform(
                X_test
            )

            if name in ["SVM", "KNN"]:

                model.fit(
                    X_train_scaled,
                    y_train
                )

                y_pred = model.predict(
                    X_test_scaled
                )

            else:

                model.fit(
                    X_train,
                    y_train
                )

                y_pred = model.predict(
                    X_test
                )

            accuracies.append(
                accuracy_score(
                    y_test,
                    y_pred
                )
            )


            print(
                f"Fold {fold}/{N_SPLITS} "
                f"Accuracy={accuracies[-1]:.4f}"
            )


            precisions.append(
                precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            )

            recalls.append(
                recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            )

            f1s.append(
                f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            )



        accuracy = np.mean(
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
        
        
        results.append({
            "Model": name,
            "SplitMode": SPLIT_MODE,
            "Accuracy": accuracy,
            "AccuracyStd": np.std(accuracies),

            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        print(
            f"Mean Accuracy : "
            f"{accuracy:.4f}"
        )

        print(
            f"Std Accuracy  : "
            f"{np.std(accuracies):.4f}"
        )



# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(results_df)

results_df.to_csv(
    "model_results.csv",
    index=False
)

print()
print("Saved:")
print("  model_results.csv")

for model_name in models.keys():

    print(
        f"  {model_name}_confusion_matrix.csv"
    )
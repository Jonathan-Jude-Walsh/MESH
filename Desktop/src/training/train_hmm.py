from hmmlearn.hmm import GaussianHMM

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.core.paths import CLASSICAL_DATASET_PATH 
from src.core.settings import SETTINGS

settings = SETTINGS

CSV_FILE = CLASSICAL_DATASET_PATH 

N_SPLITS = (
    settings["validation"]["n_splits"]
)

df = pd.read_csv(
    CSV_FILE
)

drop_cols = [
    "id",
    "label",
    "recording",
    "segment",
    "path",
    "feature_file"
]

feature_cols = [
    c
    for c in df.columns
    if c not in drop_cols
]

X = df[
    feature_cols
].fillna(0)

labels = sorted(
    df["label"].unique()
)

label_map = {
    l:i
    for i,l in enumerate(labels)
}

y = np.array([
    label_map[l]
    for l in df["label"]
])

groups = df["recording"]

gkf = GroupKFold(
    n_splits=N_SPLITS
)

scores=[]

for fold,(train_idx,test_idx) in enumerate(
    gkf.split(X,y,groups),
    start=1
):

    X_train=X.iloc[train_idx]

    X_test=X.iloc[test_idx]

    y_train=y[train_idx]

    y_test=y[test_idx]

    scaler=StandardScaler()

    X_train=scaler.fit_transform(
        X_train
    )

    X_test=scaler.transform(
        X_test
    )

    hmms={}

    for cls in np.unique(
        y_train
    ):

        model=GaussianHMM(

            n_components=4,

            covariance_type="diag",

            n_iter=100,

            random_state=42
        )

        cls_data=X_train[
            y_train==cls
        ]

        model.fit(
            cls_data
        )

        hmms[cls]=model

    preds=[]

    for sample in X_test:

        scores_cls=[]

        for cls in sorted(
            hmms.keys()
        ):

            scores_cls.append(
                hmms[cls].score(
                    sample.reshape(1,-1)
                )
            )

        preds.append(
            np.argmax(
                scores_cls
            )
        )

    acc=accuracy_score(
        y_test,
        preds
    )

    print(
        f"Fold {fold}: "
        f"{acc:.4f}"
    )

    scores.append(acc)

print()
print(
    f"Mean Accuracy: "
    f"{np.mean(scores):.4f}"
)

print(
    f"Std Accuracy : "
    f"{np.std(scores):.4f}"
)
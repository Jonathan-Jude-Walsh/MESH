from pathlib import Path

from src.config import (

    CLASSICAL_DATASET,
    CLASSICAL_DATASET_CHECKPOINT,

    DL_DATASET,
    DEMON_DATASET,
    LOFAR_DATASET,

    CLASSICAL_RESULTS,
    CNN_RESULTS,
    MOBILENET_RESULTS,
    RESNET_RESULTS,
)

# ============================================================
# Root
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[3]
)

DESKTOP_DIR = (
    ROOT_DIR / "Desktop"
)


SRC_DIR = (
    DESKTOP_DIR / "src"
)

# ============================================================
# Data
# ============================================================

DATA_DIR = (
    DESKTOP_DIR / "data"
)

DATASETS_DIR = (
    DATA_DIR / "datasets"
)

FEATURES_DIR = (
    DATA_DIR / "features"
)
CLASSICAL_DATASET_PATH = (
    DESKTOP_DIR /
    CLASSICAL_DATASET
)

CLASSICAL_CHECKPOINT_PATH = (
    DESKTOP_DIR /
    CLASSICAL_DATASET_CHECKPOINT
)

DL_DATASET_PATH = (
    DATASETS_DIR /
    DL_DATASET
)

DEMON_DATASET_PATH = (
    DATASETS_DIR /
    DEMON_DATASET
)

LOFAR_DATASET_PATH = (
    DATASETS_DIR /
    LOFAR_DATASET
)

# ============================================================
# Feature Directories
# ============================================================

MFCC_FEATURES_DIR = (
    FEATURES_DIR / "mfcc"
)

DEMON_FEATURES_DIR = (
    FEATURES_DIR / "demon"
)

LOFAR_FEATURES_DIR = (
    FEATURES_DIR / "lofar"
)

# ============================================================
# Results
# ============================================================

RESULTS_DIR = (
    DESKTOP_DIR / "results"
)

BENCHMARKS_DIR = (
    RESULTS_DIR / "benchmarks"
)

CONFUSION_DIR = (
    RESULTS_DIR / "confusion_matrices"
)

CLASSICAL_RESULTS_PATH = (
    BENCHMARKS_DIR /
    CLASSICAL_RESULTS
)

CNN_RESULTS_PATH = (
    BENCHMARKS_DIR /
    CNN_RESULTS
)

MOBILENET_RESULTS_PATH = (
    BENCHMARKS_DIR /
    MOBILENET_RESULTS
)

RESNET_RESULTS_PATH = (
    BENCHMARKS_DIR /
    RESNET_RESULTS
)

# ============================================================
# Trained Models
# ============================================================

TRAINED_MODELS_DIR = (
    DESKTOP_DIR / "trained_models"
)

# ============================================================
# Utility
# ============================================================

def ensure_directories():
    """
    Create required project directories
    if they do not already exist.
    """

    directories = [

        DATA_DIR,

        DATASETS_DIR,

        FEATURES_DIR,

        MFCC_FEATURES_DIR,

        DEMON_FEATURES_DIR,

        LOFAR_FEATURES_DIR,

        RESULTS_DIR,

        BENCHMARKS_DIR,

        CONFUSION_DIR,

        TRAINED_MODELS_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


        # ============================================================
# Dashboard Helpers
# ============================================================

def get_dataset_status():

    return {

        "deep_learning_dataset.csv":
            DL_DATASET_PATH.exists(),

        "deep_learning_dataset_demon.csv":
            DEMON_DATASET_PATH.exists(),

        "deep_learning_dataset_lofar.csv":
            LOFAR_DATASET_PATH.exists(),
    }


def get_feature_status():

    def count_files(path):

        if not path.exists():
            return 0

        return len(

            [

                f

                for f in path.rglob("*")

                if f.is_file()
            ]
        )

    return {

        "mfcc":
            count_files(
                MFCC_FEATURES_DIR
            ),

        "demon":
            count_files(
                DEMON_FEATURES_DIR
            ),

        "lofar":
            count_files(
                LOFAR_FEATURES_DIR
            ),
    }


def count_trained_models():

    if not TRAINED_MODELS_DIR.exists():

        return 0

    return len(

        [

            f

            for f in TRAINED_MODELS_DIR.rglob("*")

            if f.is_file()
        ]
    )


def list_benchmarks():

    if not BENCHMARKS_DIR.exists():

        return []

    return sorted(

        BENCHMARKS_DIR.glob(
            "*.csv"
        )
    )


def list_confusion_matrices():

    if not CONFUSION_DIR.exists():

        return []

    return sorted(

        CONFUSION_DIR.glob(
            "*.csv"
        )
    )


def get_project_status():

    return {

        "datasets":
            get_dataset_status(),

        "features":
            get_feature_status(),

        "trained_models":
            count_trained_models(),

        "benchmark_files":
            len(
                list_benchmarks()
            ),

        "confusion_matrices":
            len(
                list_confusion_matrices()
            ),
    }


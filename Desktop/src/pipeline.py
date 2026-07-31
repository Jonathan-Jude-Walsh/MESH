from pathlib import Path

import subprocess
import sys

from pathlib import Path

from src.core.result import TaskResult

from src.core.paths import (
    SRC_DIR
)

# ============================================================
# Paths
# ============================================================

DATASET_SCRIPTS = (
    SRC_DIR / "datasets"
)

FEATURE_SCRIPTS = (
    SRC_DIR / "feature_extraction"
)

TRAINING_SCRIPTS = (
    SRC_DIR / "training"
)

# ============================================================
# Utilities
# ============================================================

def run_script(
    task_name,
    script_path,
    args=None
):

    try:

        cmd = [
            sys.executable,
            str(script_path)
        ]

        if args:
            cmd.extend(args)

        result = subprocess.run(

            cmd,

            capture_output=True,

            text=True,

            check=True
        )

        return TaskResult(

            success=True,

            task_name=task_name,

            message=f"{task_name} completed successfully.",

            stdout=result.stdout
        )

    except subprocess.CalledProcessError as e:

        return TaskResult(

            success=False,

            task_name=task_name,

            message=f"{task_name} failed.",

            stdout=e.stdout,

            stderr=e.stderr
        )

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def generate_mfcc():

    return run_script(

        "MFCC Generation",

        FEATURE_SCRIPTS
        / "MFCC_generate.py"
    )


def generate_demon_lofar():

    return run_script(

        "DEMON + LOFAR Extraction",

        FEATURE_SCRIPTS
        / "extract_demon_lofar.py"
    )

# ============================================================
# DATASETS
# ============================================================

def build_mfcc_dataset():

    return run_script(

        "Build MFCC Dataset",

        DATASET_SCRIPTS
        / "build_dataset.py"
    )


def build_demon_dataset():

    return run_script(

        "Build DEMON Dataset",

        DATASET_SCRIPTS
        / "build_demon_dataset.py"
    )


def build_lofar_dataset():

    return run_script(

        "Build LOFAR Dataset",

        DATASET_SCRIPTS
        / "build_lofar_dataset.py"
    )

# ============================================================
# CLASSICAL ML
# ============================================================

def train_classical():

    return run_script(

        "Classical ML",

        TRAINING_SCRIPTS
        / "train_classical.py"
    )


def train_gmm():

    return run_script(

        "GMM",

        TRAINING_SCRIPTS
        / "train_gmm.py"
    )


def train_hmm():

    return run_script(

        "HMM",

        TRAINING_SCRIPTS
        / "train_hmm.py"
    )

# ============================================================
# DEEP LEARNING
# ============================================================

def train_cnn():

    return run_script(

        "CNN",

        TRAINING_SCRIPTS
        / "train_cnn.py"
    )


def train_mobilenet():

    return run_script(

        "MobileNet",

        TRAINING_SCRIPTS
        / "train_mobilenet.py"
    )


def train_resnet():

    return run_script(

        "ResNet",

        TRAINING_SCRIPTS
        / "train_resnet.py"
    )


def train_demon_cnn():

    return run_script(

        "DEMON + CNN",

        TRAINING_SCRIPTS
        / "train_demon_cnn.py"
    )


def train_lofar_vit():

    return run_script(

        "LOFAR + ViT",

        TRAINING_SCRIPTS
        / "train_lofar_vit.py"
    )


def train_capse_vit():

    return run_script(

        "CAPSE-ViT",

        TRAINING_SCRIPTS
        / "train_capse_vit.py"
    )


def train_catfish():

    return run_script(

        "CATFISH",

        TRAINING_SCRIPTS
        / "train_catfish.py"
    )

# ============================================================
# WORKFLOWS
# ============================================================

def run_pipeline(
    pipeline_name,
    steps
):

    results = []

    for step in steps:

        result = step()

        results.append(
            result
        )

        if not result.success:
            break

    return results


def standard_pipeline():

    return run_pipeline(

        "Standard Pipeline",

        [

            generate_mfcc,

            build_mfcc_dataset,

            train_classical,

            train_cnn,

            train_mobilenet,

            train_resnet,
        ]
    )


def demon_pipeline():

    return run_pipeline(

        "DEMON Pipeline",

        [

            generate_demon_lofar,

            build_demon_dataset,

            train_demon_cnn,
        ]
    )


def lofar_pipeline():

    return run_pipeline(

        "LOFAR Pipeline",

        [

            generate_demon_lofar,

            build_lofar_dataset,

            train_lofar_vit,
        ]
    )
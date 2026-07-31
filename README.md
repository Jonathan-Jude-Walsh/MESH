# MESH
## Modular Economical Sonobuoy Heuristic

**Low-Cost Sonobuoy for Passive Acoustic Detection and Classification of Marine Vessels**

---

## Project Overview

MESH (Modular Economical Sonobuoy Heuristic) is an open-source research project developed by **Jonathan Walsh** as part of a **420-hour Mechatronic Engineering Professional Experience** placement at the **University of Wollongong**, under the supervision of **Dr Son Lam Phung**.

The primary aim of MESH is to design, implement, and experimentally evaluate a low-cost autonomous sonobuoy capable of passively detecting and classifying marine vessels in shallow-water environments.

The project investigates the intersection of:

- Underwater Acoustics
- Passive Maritime Surveillance
- Embedded Systems
- Signal Processing
- Machine Learning
- Deep Learning
- Autonomous Sensing
- Edge AI Deployment

The objective is to balance:

- Classification Performance
- Cost
- Power Consumption
- Communications Capability
- Embedded Processing Constraints
- Deployability

within an economical and scalable sonobuoy architecture.

---

## Motivation

Traditional maritime surveillance systems are often expensive, proprietary, and inaccessible to students and researchers.

MESH explores whether advancements in:

- Raspberry Pi Computing
- Embedded Machine Learning
- Open Datasets
- Modern Signal Processing
- Low-Power Communications

can enable practical vessel classification capabilities using low-cost hardware and open-source software.

---

## Research Objectives

### Primary Objective

Develop a low-cost autonomous sonobuoy capable of:

- Passive acoustic acquisition
- Acoustic feature extraction
- Vessel classification
- Embedded deployment
- Wireless communications

### Secondary Objectives

- Investigate underwater acoustic feature extraction methods
- Compare statistical and deep learning classification approaches
- Evaluate deployment constraints on embedded hardware
- Create a modular experimentation framework
- Produce reproducible research workflows

---

## Current Capabilities

### Dataset Processing

The current framework supports:

- Audio segmentation
- Dataset generation
- Data preprocessing
- Cross-validation workflows

### Feature Extraction

Supported acoustic representations include:

- MFCC
- Mel Spectrogram
- Log-Mel Spectrogram
- DEMON
- LOFAR

### Classical Machine Learning

Implemented models include:

- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbour (KNN)
- Gaussian Mixture Model (GMM)
- Hidden Markov Model (HMM)

### Deep Learning

Implemented models include:

- CNN
- MobileNetV2
- ResNet

### Research Models In Development

- DEMON-CNN
- LOFAR-ViT
- CAPSE-ViT
- CATFISH

---

## Current Results

| Model | Accuracy |
|---------|---------:|
| Random Forest | 73.40% |
| XGBoost | 71.46% |
| KNN | 67.40% |
| SVM | 67.14% |
| ResNet | 54.91% |
| MobileNetV2 | 50.89% |

Current best performer:

**Random Forest using handcrafted acoustic features.**

These results were obtained using recording-level GroupKFold validation to prevent data leakage.

---

## Software Architecture

MESH has been refactored into a modular framework.

### Configuration

```text
config.py
```

Central source of truth for:

* Dataset filenames
* Output filenames
* Feature directory naming conventions
* Project-wide naming standards

Examples:

```python
CLASSICAL_DATASET

DL_DATASET

DEMON_DATASET

LOFAR_DATASET

CNN_RESULTS

MOBILENET_RESULTS

RESNET_RESULTS

CLASSICAL_RESULTS
```

This prevents hardcoded filenames from being scattered throughout the codebase and allows naming conventions to be managed from a single location.

***

### Path Management

```text
core/paths.py
```

Central source of truth for filesystem locations used throughout the project.

Examples:

```python
DATASETS_DIR

RESULTS_DIR

FEATURES_DIR

TRAINED_MODELS_DIR

DL_DATASET_PATH

CNN_RESULTS_PATH

MFCC_FEATURES_DIR

DEMON_FEATURES_DIR

LOFAR_FEATURES_DIR
```

All scripts use these predefined paths instead of hardcoded directories, improving portability and maintainability.

***

### Settings

```text
core/settings.py
```

Centralized storage of experiment parameters and tunable settings.

Examples include:

```python
epochs

batch_size

learning_rate

n_splits

test_size

random_seed

n_mfcc

n_fft

hop_length

dropout
```

This allows experiments to be modified without changing training or feature extraction code directly.

***

### Registry

```text
core/registry.py
```

Provides a central registry of supported machine learning and deep learning models.

Examples:

```python
Random Forest

XGBoost

KNN

SVM

GMM

HMM

CNN

MobileNetV2

ResNet

DEMON-CNN

LOFAR-ViT

CAPSE-ViT

CATFISH
```

The registry allows the graphical user interface and training pipeline to dynamically discover available models without requiring manual updates.

***

### Pipelines

```text
pipeline.py
```

Provides a unified framework for executing:

* Feature extraction
* Dataset generation
* Classical machine learning training
* Deep learning training
* Multi-stage experimental workflows

Examples:

```text
Generate Features

    ↓

Build Dataset

    ↓

Train Model

    ↓

Generate Benchmark Results
```

All pipeline actions return a structured result object, allowing consistent error handling and GUI integration.

***

### Results Management

Generated benchmark outputs are automatically written to:

```text
results/benchmarks/
```

Examples:

```text
cnn_results.csv

mobilenet_results.csv

resnet_results.csv

model_results.csv
```

Confusion matrices are written to:

```text
results/confusion_matrices/
```

Examples:

```text
RandomForest_confusion_matrix.csv

XGBoost_confusion_matrix.csv

KNN_confusion_matrix.csv

SVM_confusion_matrix.csv
```

***

### Experiment Management

The framework has been designed around reproducible experimentation.

Key principles include:

* Centralized settings
* Standardized dataset creation
* Consistent validation procedures
* Automated benchmarking
* Repeatable workflows

Current validation methods include:

```text
GroupKFold

Cross Validation

Train/Test Split
```

with an emphasis on preventing data leakage and ensuring realistic performance evaluation.

***

### Data Processing Workflow

Current workflow:

```text
Raw Audio

    ↓

Segmentation

    ↓

Feature Extraction

    ↓

Dataset Generation

    ↓

Model Training

    ↓

Cross Validation

    ↓

Benchmark Results

    ↓

Model Export
```

Supported feature types:

```text
MFCC

Mel Spectrogram

Log-Mel Spectrogram

DEMON

LOFAR
```

***

### Future Software Architecture

Planned improvements include:

#### Deployment Manager

```text
Export Models

Convert To TFLite

Package Raspberry Pi Deployments
```

#### Experiment Profiles

```text
experiments/

    default.json

    cnn_test.json

    lofar_vit.json

    catfish_experiment.json
```

#### Dashboard Improvements

```text
Feature Counts

Dataset Statistics

Trained Models

Benchmark Summaries

Automatic Refresh
```

#### GUI Refactoring

```text
src/gui/views/

    dashboard.py

    datasets.py

    training.py

    settings.py

    results.py

    deployment.py
```

This will further separate application logic from user interface code and improve maintainability.

```
```

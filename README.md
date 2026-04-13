# 🌸 Tensor Network-Based Machine Learning (Flowers Benchmark)

This repository contains the initial development of my Bachelor's Thesis (TFG), focused on applying Tensor Networks to machine learning.

This stage uses the Oxford Flowers dataset as a **preliminary benchmark** before moving to real laboratory image data.

---

## Project Overview

The main objective of this project is to explore **Tensor Networks**, specifically **Matrix Product States (MPS)**, as an alternative to classical deep learning models.

This repository represents a **development and testing phase**, where:
- Models are validated
- Architectures are tested
- Training pipelines are built

before applying them to more complex and domain-specific datasets.

---

## Current Status

⚠️ Work in progress

- ✅ Data pipeline implemented (Oxford Flowers dataset)
- ✅ Classical CNN model implemented and trained
- ✅ Initial MPS structure defined
- 🔄 Integration of MPS into training pipeline (ongoing)
- 🔄 Migration to laboratory datasets (future stage)

---

## Project Structure
├── Flower_DL.py # Main training script

├── utils.py # Dataset, models, training & evaluation functions


- Custom dataset class implemented
- Classical baselines:
  - Linear classifier
  - CNN
- Tensor Network model:
  - MPS (Matrix Product State)

The MPS architecture is defined in `utils.py`

---

## Models

### 🔹 Classical Baselines
- Fully connected network
- Convolutional Neural Network (CNN)

Used as reference for performance comparison.

### 🔹 Tensor Network Model
- Matrix Product State (MPS)
- Implemented with `tensorkrowch`
- Modular and scalable architecture

Currently under development and not fully integrated into training.

---

## Training Pipeline

- Image preprocessing (resize, normalization)
- Dataset split: train / validation / test
- Training with cross-entropy loss
- Evaluation on validation and test sets

---

## Future Work

- Full integration of MPS into training loop
- Hyperparameter optimization (bond dimension, structure)
- Benchmark vs classical deep learning models
- Transition to laboratory image datasets

---

## Author

**Manuel Arenas Sánchez**  
BSc Physics — Final Year Project

---

## Notes

This repository will be continuously updated as the project evolves.

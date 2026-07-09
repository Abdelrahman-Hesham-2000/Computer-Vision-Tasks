# Eye Disease Classification

## Overview

This project focuses on classifying retinal fundus images into *8 eye disease classes* using Deep Learning.

Two approaches were implemented and compared:

- Custom CNN built from scratch.
- Transfer Learning using EfficientNetB0.

The project includes data preprocessing, model training, feature extraction, fine-tuning, and model evaluation.

---

## Dataset

- Multi-class Eye Disease Dataset
- 8 Classes
- Image Size: *256 × 256*
- Train / Validation / Test Split

---

## Data Preprocessing

- Image resizing
- Normalization
- Data Augmentation
  - Rotation
  - Width Shift
  - Height Shift
  - Zoom
  - Horizontal Flip

---

## Models

### Custom CNN

- 5 Convolution Blocks
- Batch Normalization
- Max Pooling
- Global Average Pooling
- Dense Layers
- Dropout

### EfficientNetB0

The pretrained EfficientNetB0 model was trained in two stages:

- Feature Extraction
- Fine-Tuning (last 30 layers)

---

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Files

- Eye_Disease_Classification.ipynb → Complete notebook
- requirements.txt → Required libraries


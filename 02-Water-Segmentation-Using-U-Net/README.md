# Water Segmentation Using U-Net

## Overview

This project performs semantic segmentation of water bodies in satellite imagery using the U-Net deep learning architecture implemented with PyTorch.

The model learns to classify every pixel in an image as either:

- Water
- Non-water

This project covers the complete deep learning pipeline including data preprocessing, model training, evaluation, and prediction visualization.

---

## Dataset

The dataset consists of:

- RGB satellite images
- Binary segmentation masks

The dataset is divided into:

- Training set
- Validation set

---

## Project Pipeline

1. Import libraries
2. Load images and masks
3. Visualize samples
4. Split the dataset
5. Create a custom PyTorch Dataset
6. Build the U-Net architecture
7. Train the model
8. Evaluate using Dice Score and IoU
9. Predict segmentation masks
10. Visualize predictions

---

## Model Architecture

The project uses the U-Net architecture consisting of:

- Encoder
- Bottleneck
- Decoder
- Skip Connections
- Final 1×1 Convolution Layer

---

## Loss Function

Binary Cross Entropy Loss (BCE)

---

## Evaluation Metrics

- Dice Score
- Intersection over Union (IoU)

---

## Training

Optimizer:

- Adam

The training process records:

- Training Loss
- Validation Loss
- Dice Score
- IoU Score


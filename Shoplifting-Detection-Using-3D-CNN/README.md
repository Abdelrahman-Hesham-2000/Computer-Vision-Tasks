Shoplifting Detection Using 3D CNN

A video classification project for detecting shoplifting behavior using a custom 3D Convolutional Neural Network (3D CNN) implemented with PyTorch.

The model classifies videos into two classes:

Non-Shoplifting

Shoplifting

Project Overview

The project processes videos as short spatiotemporal clips. Instead of treating each video frame independently, the 3D CNN learns spatial features from individual frames while also modeling temporal information across consecutive sampled frames.

Pipeline

Video Dataset
     │
     ├── Dataset inspection
     ├── Duplicate-content check
     ├── Train / Validation / Test split
     │
     ▼
Uniform frame sampling
     │
     ▼
Resize frames to 112 × 112
     │
     ▼
Normalize pixel values to [0, 1]
     │
     ▼
32 frames per video
     │
     ▼
3D CNN
     │
     ▼
Binary classification
     │
     ├── Non-Shoplifting
     └── Shoplifting

Dataset

The dataset contains 855 videos:

Class

Videos

Percentage

Non-Shoplifting

531

62.11%

Shoplifting

324

37.89%

Total

855

100%

The videos contain between 75 and 1,850 frames, with an average of approximately 331 frames and a median of 325 frames.

The dataset is not included in this repository because of its size. The notebook downloads the dataset from Google Drive when executed in the Kaggle environment.

Data Preprocessing

For every video:

The video is opened using OpenCV.

32 frames are sampled uniformly across the full video.

Each frame is converted from BGR to RGB.

Frames are resized to 112 × 112.

Pixel values are normalized to [0, 1].

The tensor is rearranged from (T, H, W, C) to (C, T, H, W) for PyTorch 3D convolution.

Videos that cannot be read are represented by zero-filled frames, while missing sampled frames can be filled using the last valid frame.

Data Augmentation

The augmentation function is currently disabled and returns the original frames unchanged.

An earlier augmentation implementation is kept commented in the notebook and includes:

Horizontal flipping

Random resized crop

Brightness adjustment

Contrast adjustment

Saturation adjustment

Train / Validation / Test Split

The dataset is split using stratified sampling:

70% Training

15% Validation

15% Test

The split is stratified by class label to preserve the class distribution.

Model Architecture

The project uses a custom 3D CNN built with PyTorch.

Feature Extractor

Conv3D: 3 → 32

GroupNorm

ReLU

Spatial MaxPool3D

Conv3D: 32 → 64

GroupNorm

ReLU

Spatial MaxPool3D

Conv3D: 64 → 128

GroupNorm

ReLU

MaxPool3D

Conv3D: 128 → 256

GroupNorm

ReLU

Dropout3D

Adaptive Average Pooling

Classifier

Linear: 256 → 256

ReLU

Dropout

Linear: 256 → 128

ReLU

Dropout

Linear: 128 → 2

Kaiming initialization is used for convolutional and linear layers.

Training Configuration

Parameter

Value

Image size

112 × 112

Frames per video

32

Number of classes

2

Batch size

8

Epochs

20

Optimizer

Adam

Learning rate

3e-4

Weight decay

1e-5

LR scheduler

ReduceLROnPlateau

LR factor

0.5

LR patience

3

Gradient clipping

max norm 5.0

Random seed

42

A WeightedRandomSampler is used during training to reduce the effect of class imbalance.

Results

The final test evaluation produced:

Metric

Score

Accuracy

62.02%

Precision

0.00

Recall

0.00

F1-score

0.00

The classification report shows that the final model predicted the Non-Shoplifting class while failing to correctly identify the Shoplifting class.

This means that although the accuracy is around 62%, the model is not suitable yet for reliable shoplifting detection.

Important Training Observation

The best validation F1-score reached 0.5455 at epoch 4, but the final test evaluation in the notebook is performed using the model state from the end of epoch 20.

For a proper final evaluation, the saved best_3dcnn_model.pth checkpoint should be loaded before evaluating the test set.

Also, class weights are calculated in the notebook, but the current CrossEntropyLoss is created without passing those weights. This is a potential improvement for the next experiment.

Hardware

The notebook was executed with:

2 × NVIDIA Tesla T4 GPUs

Approximately 14.56 GB GPU memory per GPU

CUDA 12.8

The model automatically uses CUDA when available.

Installation

pip install -r requirements.txt

Running the Project

The main experiment is provided as a Jupyter Notebook:

shoplifting-detection-using-3d-cnn.ipynb

The notebook was designed for a Kaggle environment and downloads the dataset from Google Drive.

If running outside Kaggle, update the dataset download and extraction paths accordingly.

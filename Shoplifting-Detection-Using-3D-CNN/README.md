# 🎥 Shoplifting Detection Using 3D CNN

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-3D%20CNN-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Computer%20Vision-Video%20Classification-blue" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Dataset-855%20Videos-green" alt="Dataset">
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-orange" alt="Task">
</p>

<p align="center">
  <b>A deep learning approach for detecting shoplifting behavior from video using a custom 3D Convolutional Neural Network.</b>
</p>

---

## 📌 Overview

This project focuses on **video-based shoplifting detection** using a custom **3D Convolutional Neural Network (3D CNN)** implemented with PyTorch.

Unlike traditional image classification, where each frame is processed independently, a 3D CNN learns from both:

* 🖼️ **Spatial information** — visual features within each frame
* ⏱️ **Temporal information** — changes and actions across consecutive frames

The model performs binary classification:

> **Non-Shoplifting vs Shoplifting**

---

## 🚀 Project Pipeline

```text
                    ┌─────────────────────┐
                    │     Video Dataset   │
                    │      855 Videos     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Video Inspection  │
                    │ & Duplicate Check   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Uniform Frame       │
                    │ Sampling            │
                    │      32 Frames      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Resize + Normalize  │
                    │    112 × 112        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      3D CNN         │
                    │  Spatial + Temporal │
                    │      Features       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Classification    │
                    │                     │
                    │ Non-Shoplifting     │
                    │ Shoplifting         │
                    └─────────────────────┘
```

---

# 📊 Dataset

The dataset contains **855 videos** belonging to two classes.

| Class              |  Videos | Percentage |
| :----------------- | ------: | ---------: |
| 🟢 Non-Shoplifting |     531 |     62.11% |
| 🔴 Shoplifting     |     324 |     37.89% |
| **Total**          | **855** |   **100%** |

### 🎞️ Video Statistics

| Statistic      | Value  |
| :------------- | :----- |
| Minimum frames | 75     |
| Maximum frames | 1,850  |
| Average frames | 331.42 |
| Median frames  | 325    |

### 📥 Download Dataset

The complete dataset is available on Google Drive:

<p align="center">
  <a href="https://drive.google.com/file/d/1KCKfyIGbQi8a7bIYta3LM8dFStxVzVX-/view">
    <img src="https://img.shields.io/badge/Download-Dataset-blue?style=for-the-badge&logo=googledrive&logoColor=white" alt="Download Dataset">
  </a>
</p>

> **Note:** The dataset is not included in this repository because of its large size.

---

# 🔍 Data Inspection

Before training, the dataset was inspected for:

* Missing values
* Duplicate video paths
* Class distribution
* Video length distribution
* Potential duplicate video content

An MD5-based check was also performed on selected videos to verify whether identical video content existed under different filenames.

---

# ⚙️ Preprocessing

Each video goes through the following preprocessing pipeline.

### 1. Uniform Frame Sampling

**32 frames** are uniformly sampled across the entire video.

```text
Video
│
├── Frame 1
├── Frame 2
├── ...
├── Frame 32
│
└── Sampled uniformly across video duration
```

### 2. Color Conversion

OpenCV reads frames in BGR format, so each frame is converted to RGB.

### 3. Resizing

Every frame is resized to:

```text
112 × 112 × 3
```

### 4. Normalization

Pixel values are normalized from:

```text
[0, 255] → [0, 1]
```

### 5. Tensor Format

The video tensor is converted from:

```text
(T, H, W, C)
```

to PyTorch's 3D CNN format:

```text
(C, T, H, W)
```

Therefore, each input video becomes:

```text
3 × 32 × 112 × 112
```

---

# ✂️ Dataset Split

A **stratified train/validation/test split** was used:

| Split      | Percentage |
| :--------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

Stratification preserves the class distribution across all three subsets.

---

# 🧠 Model Architecture

A custom **3D CNN** was developed from scratch using PyTorch.

The network extracts both spatial and temporal information from the video clips.

## Feature Extractor

```text
Input
3 × 32 × 112 × 112
        │
        ▼
┌──────────────────────┐
│ Conv3D  3 → 32       │
│ GroupNorm            │
│ ReLU                 │
│ MaxPool3D            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Conv3D  32 → 64      │
│ GroupNorm            │
│ ReLU                 │
│ MaxPool3D            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Conv3D  64 → 128     │
│ GroupNorm            │
│ ReLU                 │
│ MaxPool3D             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Conv3D  128 → 256    │
│ GroupNorm            │
│ ReLU                 │
│ Dropout3D            │
└──────────┬───────────┘
           │
           ▼
Adaptive Average Pooling
           │
           ▼
        256 Features
```

## Classification Head

```text
256
 │
 ▼
Linear 256 → 256
 │
 ▼
ReLU
 │
 ▼
Dropout 0.4
 │
 ▼
Linear 256 → 128
 │
 ▼
ReLU
 │
 ▼
Dropout 0.3
 │
 ▼
Linear 128 → 2
 │
 ▼
┌───────────────────────┐
│ 0 → Non-Shoplifting   │
│ 1 → Shoplifting       │
└───────────────────────┘
```

### Weight Initialization

Kaiming initialization is used for:

* `Conv3D` layers
* `Linear` layers

---

# 🏋️ Training Configuration

| Parameter         | Value             |
| :---------------- | :---------------- |
| Framework         | PyTorch           |
| Input Size        | 112 × 112         |
| Frames / Video    | 32                |
| Batch Size        | 8                 |
| Epochs            | 20                |
| Optimizer         | Adam              |
| Learning Rate     | 3 × 10⁻⁴          |
| Weight Decay      | 1 × 10⁻⁵          |
| LR Scheduler      | ReduceLROnPlateau |
| LR Factor         | 0.5               |
| LR Patience       | 3                 |
| Gradient Clipping | 5.0               |
| Random Seed       | 42                |

A `WeightedRandomSampler` was used during training to reduce the effect of class imbalance.

---

# 📈 Results

## Test Performance

| Metric       |      Score |
| :----------- | ---------: |
| **Accuracy** | **62.02%** |
| Precision    |       0.00 |
| Recall       |       0.00 |
| F1-score     |       0.00 |

### Classification Report

| Class           | Precision | Recall | F1-score |
| :-------------- | --------: | -----: | -------: |
| Non-Shoplifting |      0.62 |   1.00 |     0.77 |
| Shoplifting     |      0.00 |   0.00 |     0.00 |

---

# ⚠️ Important Result

Although the model achieved approximately **62% accuracy**, the result should **not** be interpreted as successful shoplifting detection.

The model failed to correctly identify the Shoplifting class:

```text
Shoplifting Recall = 0.00
Shoplifting F1     = 0.00
```

The dataset contains approximately **62% Non-Shoplifting videos**, meaning that predicting the majority class can already produce an accuracy close to 62%.

Therefore:

> **Accuracy alone is not sufficient to evaluate this model.**

For this task, metrics such as **Recall, F1-score, and the Confusion Matrix** are much more informative.

---

# 📉 Training Observation

The model showed a tendency toward majority-class predictions during training.

The best validation F1-score observed during training was approximately:

```text
Validation F1 = 0.5455
```

This occurred around **Epoch 4**.

However, the final test evaluation was performed using the model state at the end of training rather than reloading the best saved checkpoint.

This is an important area for improvement in the next experiment.

---

# 🔬 Error Analysis

The current results suggest several possible limitations.

### 1. Class Imbalance

The dataset contains more Non-Shoplifting videos than Shoplifting videos.

### 2. Training From Scratch

The model is a custom 3D CNN trained from scratch, which can make learning robust video representations more difficult with a relatively small dataset.

### 3. Limited Temporal Sampling

Only 32 frames are sampled from each video.

Short actions occurring between sampled frames may not be captured sufficiently.

### 4. No Active Augmentation

The final experiment does not apply augmentation.

The notebook contains an augmentation implementation, but it is currently disabled.

### 5. Best Checkpoint Not Used During Testing

The training loop saves:

```text
best_3dcnn_model.pth
```

based on the best validation F1-score.

However, the final test evaluation should reload this checkpoint before calculating the final metrics.

---

# 🚀 Future Improvements

The following experiments would be valuable:

### 🔹 Better Checkpoint Evaluation

Load the best validation checkpoint before testing.

```python
model.load_state_dict(
    torch.load("best_3dcnn_model.pth",
               map_location=device)
)
```

### 🔹 Class-Weighted Loss

The notebook calculates class weights but currently does not pass them to the loss function.

A future experiment can use:

```python
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)
```

### 🔹 Video Augmentation

Introduce temporally consistent augmentation such as:

* Horizontal flipping
* Random cropping
* Brightness variation
* Contrast variation
* Saturation variation

### 🔹 Temporal Experiments

Compare:

```text
16 frames
32 frames
64 frames
```

and investigate different temporal sampling strategies.

### 🔹 Pretrained Video Models

Compare the custom 3D CNN against pretrained architectures to determine whether transfer learning provides better spatiotemporal representations.

### 🔹 Additional Metrics

Evaluate:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Balanced Accuracy

```

> The dataset should **not** be committed to the repository.

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Shoplifting-Detection-3D-CNN.git
```

Move into the project directory:

```bash
cd Shoplifting-Detection-3D-CNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

The complete experiment is available in:

```text
shoplifting-detection-using-3d-cnn.ipynb
```

The notebook was developed for a **Kaggle GPU environment** and downloads the dataset from Google Drive.

Open the notebook in Kaggle or Jupyter and execute the cells sequentially.

---

# 🖥️ Hardware

The experiment was executed using:

```text
GPU: 2 × NVIDIA Tesla T4
CUDA: 12.8
Framework: PyTorch
```

The notebook automatically detects CUDA and uses the available GPU when possible.

---

# 📌 Key Takeaways

This project demonstrates an end-to-end video classification pipeline using a custom 3D CNN:

```text
Video
 ↓
Frame Sampling
 ↓
Preprocessing
 ↓
3D Convolution
 ↓
Spatiotemporal Feature Extraction
 ↓
Binary Classification
```

The experiment establishes a baseline for shoplifting detection and highlights the challenges of video classification with limited data and class imbalance.

The current model is **not production-ready**, but it provides a foundation for future experiments involving better sampling, augmentation, class-weighted learning, checkpoint selection, and pretrained video models.

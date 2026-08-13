# 🛒 Shoplifting Detection Using R3D-18

A deep learning video classification project for detecting **shoplifting behavior** from surveillance videos using a **pretrained R3D-18 3D Convolutional Neural Network**.

The model analyzes temporal information across video frames and classifies each video into one of two classes:

* **Non-Shoplifting**
* **Shoplifting**

---

## 📌 Project Overview

Traditional image classification models analyze individual frames independently. However, shoplifting is a **temporal behavior**, meaning that the sequence of actions across multiple frames is important.

This project uses **R3D-18**, a 3D CNN architecture designed for video understanding, to learn both:

* Spatial information from individual frames
* Temporal information across consecutive frames

The R3D-18 backbone is initialized with pretrained **Kinetics weights** and fine-tuned for binary shoplifting classification.

---

## 🧠 Model Architecture

The project uses:

**R3D-18 (3D ResNet-18)**

The original classification layer is replaced with:

```text
R3D-18 Backbone
      ↓
Dropout (0.4)
      ↓
Linear Layer
      ↓
2 Classes
```

The model uses pretrained `R3D_18_Weights.DEFAULT` from TorchVision.

---

## 📂 Dataset

The dataset contains surveillance videos divided into two classes:

```text
Shop DataSet/
├── non shop lifters/
│   └── *.mp4
│
└── shop lifters/
    └── *.mp4
```

### Dataset Statistics

| Class           |  Videos |
| --------------- | ------: |
| Non-Shoplifting |     313 |
| Shoplifting     |     324 |
| **Total**       | **637** |

The original dataset contained **855 videos**. Exact duplicate videos were detected using MD5 hashing and removed before training.

The duplicate-checking pipeline found:

* 855 videos initially
* 436 videos involved in exact duplicate groups
* 0 cross-class exact duplicates
* 637 unique videos after deduplication

---

## 🎥 Video Preprocessing

Each video is processed using the following pipeline:

1. Read the video using OpenCV.
2. Uniformly sample **128 frames** across the entire video.
3. Convert frames from BGR to RGB.
4. Resize each frame to **112 × 112**.
5. Normalize pixel values to `[0, 1]`.
6. Apply training-time video augmentation.
7. Rearrange the tensor to:

```text
(C, T, H, W)
```

where:

```text
C = 3
T = 128
H = 112
W = 112
```

The final batch input to R3D-18 is:

```text
(B, 3, 128, 112, 112)
```

---

## 🔄 Data Augmentation

Augmentation is applied only during training.

This helps improve the model's ability to generalize to different surveillance conditions and video variations.

---

## ⚖️ Class Imbalance

The dataset contains slightly different numbers of videos between the two classes.

To address this, balanced class weights are calculated using:

```python
compute_class_weight()
```

and incorporated into:

```python
nn.CrossEntropyLoss(weight=class_weights)
```

This prevents the model from becoming biased toward the majority class.

---

## 🏋️ Training Configuration

| Parameter         |             Value |
| ----------------- | ----------------: |
| Model             |            R3D-18 |
| Pretrained        |               Yes |
| Image Size        |         112 × 112 |
| Frames per Video  |               128 |
| Number of Classes |                 2 |
| Batch Size        |                 2 |
| Epochs            |                10 |
| Learning Rate     |              1e-4 |
| Weight Decay      |              1e-5 |
| Optimizer         |              Adam |
| Scheduler         | ReduceLROnPlateau |
| LR Factor         |               0.5 |
| LR Patience       |                 3 |
| Random Seed       |                42 |

Gradient clipping is also used with:

```python
max_norm=5.0
```

---

## 📊 Evaluation

The model is evaluated on a held-out test set using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

Example evaluation:

```text
Test Accuracy
Precision
Recall
F1-score
```

The confusion matrix is used to analyze errors between:

```text
Non-Shoplifting
Shoplifting
```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* TorchVision
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* tqdm
* Google Drive / gdown
* Kaggle GPU

---

## 📁 Project Structure

```text
Shoplifting-Detection-R3D18/
│
├── shoplifting-detection-using-r3d-18.ipynb
├── README.md
├── requirements.txt
│
└── models/
    └── best_r3d18_train_loss_model.pth
```

> **Note:** The dataset and trained model weights are not included in the GitHub repository because of their large file sizes.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Shoplifting-Detection-R3D18.git

cd Shoplifting-Detection-R3D18
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Run:

```text
shoplifting-detection-using-r3d-18.ipynb
```

The notebook was developed and tested in a Kaggle GPU environment.

---

## 💾 Dataset

The dataset is downloaded from Google Drive inside the notebook using `gdown`.

The notebook automatically downloads:

```text
dataset.zip
```

and extracts it into:

```text
/kaggle/working/Shop DataSet
```

You can replace the dataset download section with your own dataset path if running locally.

---

## ⚡ Hardware

The training notebook was executed using:

```text
2 × NVIDIA Tesla T4
14.56 GB GPU Memory per GPU
CUDA 12.8
```

The code automatically detects CUDA and uses GPU when available.

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## 🔬 Key Features

### Duplicate Detection

The project includes a dedicated duplicate-detection pipeline using:

* MD5 hashing
* Video metadata comparison
* Frame-based visual comparison
* dHash similarity

This helps prevent duplicated videos from affecting the training process.

### Temporal Video Modeling

Unlike standard 2D CNNs, R3D-18 processes video clips using 3D convolutions, allowing the model to learn temporal patterns in addition to spatial features.

### Transfer Learning

The model uses a pretrained R3D-18 backbone with Kinetics pretrained weights, reducing the amount of training required compared with training a 3D CNN completely from scratch.

---

## 📌 Results

The notebook evaluates the final model using a held-out test set and reports:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

The exact results are available in the notebook output.

---

## 🔮 Future Improvements

Possible improvements include:

* Training for more epochs
* Using a larger batch size when GPU memory allows
* Fine-tuning learning rates
* Stronger temporal augmentation
* Using longer or multiple video clips per sample
* Testing other video architectures such as:

  * R(2+1)D
  * MC3
  * Video Swin Transformer
  * X3D
* Using cross-validation
* Adding real-time webcam/video inference
* Deploying the model as an API

---

## 👨‍💻 Author

**Abdelrahman Hesham**

Machine Learning Engineer | Computer Vision

---

## ⭐ Acknowledgment

This project uses the R3D-18 architecture and pretrained weights provided by **TorchVision**.

If you find this project useful, consider giving the repository a ⭐.


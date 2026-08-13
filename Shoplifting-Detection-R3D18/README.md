# 🛒 Shoplifting Detection Using R3D-18

A deep learning video classification project for detecting shoplifting behavior from surveillance videos using a pretrained R3D-18 3D Convolutional Neural Network.

The model classifies surveillance videos into two categories:

- 🛒 Shoplifting
- ✅ Non-Shoplifting

---

## 📌 Project Overview

Shoplifting is a temporal behavior, which means that analyzing a single image or frame may not be sufficient to correctly identify the activity.

To capture both spatial and temporal information, this project uses R3D-18, a 3D CNN architecture based on ResNet.

The model is initialized with pretrained Kinetics weights and fine-tuned for binary video classification.

### Why R3D-18?

Unlike traditional 2D CNNs that process individual frames, 3D CNNs use 3D convolutions to learn:

- Spatial features from video frames
- Temporal patterns across consecutive frames

This makes R3D-18 suitable for video-based action recognition tasks such as shoplifting detection.

---

## 🧠 Model Architecture

The project uses R3D-18 (3D ResNet-18).

The pretrained classification head is replaced with a binary classification layer.

Input Video → Frame Sampling → Preprocessing → R3D-18 Backbone → Dropout → Fully Connected Layer → 2 Classes

Classes:

- Non-Shoplifting
- Shoplifting

The R3D-18 model is initialized using pretrained Kinetics weights.

---

## 📂 Dataset

The dataset consists of surveillance videos divided into two classes:

| Class | Description |
|---|---|
| 🛒 Shoplifting | Videos containing shoplifting behavior |
| ✅ Non-Shoplifting | Videos without shoplifting behavior |

The dataset is hosted externally on Google Drive because of its large size and is not included in this GitHub repository.

### 📥 Download Dataset

[Download the Shoplifting Dataset](https://drive.google.com/file/d/1KCKfyIGbQi8a7bIYta3LM8dFStxVzVX-/view)

Make sure the Google Drive file is accessible to users with the link.

---

## 🎥 Video Preprocessing

Each video goes through a preprocessing pipeline before being passed to the model.

### Processing Steps

1. Read the video using OpenCV.
2. Uniformly sample frames from the video.
3. Convert frames from BGR to RGB.
4. Resize frames to 112 × 112.
5. Normalize pixel values.
6. Apply training augmentation.
7. Rearrange the tensor into the format required by R3D-18.

The input tensor format is (B, C, T, H, W).

Where:

- B = Batch Size
- C = 3 RGB Channels
- T = Number of Frames
- H = 112
- W = 112

Example input shape:

(B, 3, 128, 112, 112)

---

## 🔄 Data Augmentation

Data augmentation is applied during training to improve the model's ability to generalize to different surveillance conditions.

Augmentation helps the model become more robust to variations in:

- Camera position
- Object position
- Lighting
- Motion
- Video appearance

---

## ⚖️ Class Imbalance

The dataset contains a slightly different number of samples between the two classes.

To reduce the effect of class imbalance, class weights are calculated and used with the Cross Entropy Loss function.

The loss function uses weighted Cross Entropy Loss to give appropriate importance to each class during training.

---

## 🏋️ Training Configuration

| Parameter | Value |
|---|---|
| Model | R3D-18 |
| Pretrained | Yes |
| Pretrained Dataset | Kinetics |
| Number of Classes | 2 |
| Frames per Video | 128 |
| Frame Size | 112 × 112 |
| Batch Size | 2 |
| Epochs | 10 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| Optimizer | Adam |
| Loss Function | Weighted Cross Entropy |
| LR Scheduler | ReduceLROnPlateau |
| Gradient Clipping | Yes |
| Random Seed | 42 |

---

## 📊 Evaluation

The trained model is evaluated using a separate test set.

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

The confusion matrix helps analyze how well the model distinguishes between Non-Shoplifting and Shoplifting.

The detailed evaluation results and visualizations are available inside the Jupyter Notebook.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm
- gdown

---

## 📁 Project Structure

Shoplifting-Detection-R3D18/
├── shoplifting-detection-using-r3d-18.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── models/
    └── best_r3d18_train_loss_model.pth

Large files such as the dataset and trained model weights should not be committed directly to GitHub.

---

## 🚀 How to Run

### 1. Clone the Repository

Clone the repository and navigate to the project directory.

git clone https://github.com/YOUR_USERNAME/Shoplifting-Detection-R3D18.git

cd Shoplifting-Detection-R3D18

### 2. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

### 3. Download the Dataset

Download the dataset from Google Drive:

[Shoplifting Dataset](https://drive.google.com/file/d/1KCKfyIGbQi8a7bIYta3LM8dFStxVzVX-/view)

Then extract the dataset and update the dataset path inside the notebook if necessary.

### 4. Run the Notebook

Open the Jupyter Notebook:

shoplifting-detection-using-r3d-18.ipynb

Run the notebook cells sequentially.

The project was designed to take advantage of GPU acceleration for training.

---

## ⚡ Hardware

Training deep video models such as R3D-18 requires significant GPU memory.

The notebook was developed using GPU acceleration.

CUDA is automatically detected using PyTorch.

If CUDA is unavailable, the model can fall back to CPU, although training will be significantly slower.

---

## 🔍 Duplicate Detection

To avoid data leakage and duplicated samples, the dataset was checked for duplicate videos.

The project uses hashing-based methods to identify exact duplicate files before training.

This helps ensure that duplicate videos do not unnecessarily appear in different dataset splits.

---

## 🔬 Transfer Learning

Instead of training R3D-18 completely from scratch, the model uses pretrained weights.

The pretrained backbone has already learned useful visual and temporal representations from the Kinetics dataset.

The model is then adapted to the specific task of:

Video → Shoplifting / Non-Shoplifting

This approach provides a strong initialization compared with training a 3D CNN completely from scratch.

---

## 📈 Results

The model is evaluated using the held-out test set.

Performance is reported using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

The detailed results and visualizations are available inside the Jupyter Notebook.

---

## 🔮 Future Improvements

Several improvements can be explored in future versions of the project:

- Train for more epochs
- Experiment with different learning rates
- Increase the number of sampled frames
- Experiment with different frame resolutions
- Use stronger temporal augmentation
- Perform more extensive hyperparameter tuning
- Compare R3D-18 with other video architectures
- Experiment with R(2+1)D
- Experiment with MC3
- Experiment with X3D
- Experiment with Video Transformers
- Implement real-time video inference
- Deploy the trained model as an API or web application

---

## 📚 Project Workflow

Dataset → Duplicate Detection → Dataset Splitting → Video Frame Sampling → Preprocessing → Data Augmentation → Pretrained R3D-18 → Training → Evaluation

---

## ⭐ Acknowledgment

This project uses the R3D-18 architecture and pretrained video classification weights provided through TorchVision.

If you find this project useful, consider giving the repository a ⭐.

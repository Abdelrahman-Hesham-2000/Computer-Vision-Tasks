# Image Captioning From Scratch

A deep learning project for generating natural language descriptions of images using a **CNN-based image encoder** and a **Transformer-style decoder with Cross-Attention**.

The entire model is trained **from scratch** using PyTorch without pretrained models or pretrained weights.

## 🚀 Project Overview

Image captioning is a multimodal deep learning task that combines:

* **Computer Vision** for understanding image content.
* **Natural Language Processing** for generating descriptive text.

This project follows an Encoder-Decoder architecture:

```text
Input Image
     │
     ▼
CNN Vision Encoder
     │
     ▼
Visual Tokens
     │
     ▼
Transformer Decoder
(Self-Attention + Cross-Attention)
     │
     ▼
Generated Caption
```

The CNN encoder extracts visual features from the input image and converts them into a sequence of visual tokens.

The Transformer-style decoder then generates the caption token by token while attending to:

1. Previous caption tokens using **Masked Self-Attention**.
2. Image features using **Cross-Attention**.

---

## 🧠 Model Architecture

### 1. CNN Image Encoder

The image encoder is a lightweight CNN built from scratch.

The architecture gradually increases the number of channels:

```text
Input Image
    │
    ▼
3 Channels
    │
    ▼
64 Channels
    │
    ▼
128 Channels
    │
    ▼
256 Channels
    │
    ▼
512-Dimensional Visual Features
```

Each CNN block contains:

```text
Conv2D
   ↓
Batch Normalization
   ↓
GELU
   ↓
Conv2D
   ↓
Batch Normalization
   ↓
GELU
```

The final CNN feature map is converted into a sequence of visual tokens.

---

### 2. Transformer-Style Decoder

The decoder consists of multiple layers.

Each decoder layer contains:

* Masked Self-Attention
* Cross-Attention
* Feed-Forward Network
* Layer Normalization
* Residual Connections
* Dropout

The decoder uses **causal masking** to prevent the model from seeing future words while generating captions.

```text
Caption Tokens
      │
      ▼
Masked Self-Attention
      │
      ▼
Cross-Attention ◄──── Visual Tokens from CNN Encoder
      │
      ▼
Feed Forward Network
      │
      ▼
Vocabulary Predictions
```

---

## ⚙️ Model Configuration

| Parameter               |     Value |
| ----------------------- | --------: |
| Image Size              | 192 × 192 |
| Batch Size              |        64 |
| Epochs                  |        15 |
| Early Stopping Patience |         3 |
| Learning Rate           |      5e-4 |
| Model Dimension         |       512 |
| Attention Heads         |         8 |
| Decoder Layers          |         6 |
| Dropout                 |       0.1 |
| Maximum Vocabulary Size |    12,000 |
| Minimum Word Frequency  |         2 |
| Validation Images       |       500 |
| Test Images             |       500 |

---

## 📊 Dataset Split

The dataset is split based on **unique images**.

```text
Dataset
   │
   ├── Training Set
   │
   ├── Validation Set
   │      └── 500 Images
   │
   └── Test Set
          └── 500 Images
```

The validation set is used to monitor training and select the best model.

The test set remains completely independent and is only used for the final evaluation.

---

## 🏋️ Training

The model is trained using:

* **AdamW Optimizer**
* **CrossEntropyLoss**
* **Cosine Annealing Learning Rate Scheduler**
* **Mixed Precision Training**
* **Early Stopping**

Padding tokens are ignored during loss calculation.

---

## 📈 Evaluation Metrics

The final model is evaluated using the following image captioning metrics:

### BLEU-4

Measures the similarity between generated captions and reference captions using n-gram precision.

### METEOR

Evaluates caption similarity while considering factors such as word matching and linguistic variations.

### CIDEr

Measures the similarity between generated captions and multiple human-written reference captions.

The independent test set contains **500 images** and is used only for final evaluation.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Pillow
* NLTK
* tqdm

---

## 📂 Project Structure

```text
Image-Captioning-From-Scratch/
│
├── Image_Captioning_From_Scratch.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate to the project directory:

```bash
cd YOUR_REPOSITORY
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Open the Jupyter Notebook:

```bash
jupyter notebook Image_Captioning_From_Scratch.ipynb
```

The notebook will:

1. Detect the available computing device.
2. Load and process the image-caption dataset.
3. Build the vocabulary.
4. Split the dataset into training, validation, and test sets.
5. Build the CNN Encoder and Transformer Decoder.
6. Train the model from scratch.
7. Select the best model using validation performance.
8. Generate captions for test images.
9. Evaluate the model using BLEU-4, METEOR, and CIDEr.

---

## 🔥 Key Features

* CNN encoder trained completely from scratch.
* Transformer-style decoder with Cross-Attention.
* Multi-head Self-Attention.
* Causal masking for autoregressive caption generation.
* Positional embeddings.
* Weight tying between token embeddings and output projection.
* Mixed precision training.
* Cosine learning rate scheduling.
* Early stopping.
* Independent validation and test sets.
* Evaluation using BLEU-4, METEOR, and CIDEr.

---

## 🎯 Future Improvements

Possible improvements include:

* Using pretrained vision encoders such as ResNet or EfficientNet.
* Using Vision Transformers.
* Beam Search for better caption generation.
* Larger datasets.
* Larger vocabulary sizes.
* Training for more epochs.
* Using pretrained language models as decoders.
* Applying data augmentation.
* Experimenting with different encoder and decoder architectures.

⭐ If you found this project useful, feel free to star the repository!


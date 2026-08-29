# Image Captioning with a Frozen Pretrained ViT-B/16

A deep learning project for generating natural language descriptions of images using a **frozen, pretrained ViT-B/16 image encoder** and a **Transformer-style decoder with Cross-Attention**.

Only the decoder is trained. The ViT-B/16 encoder keeps its **pretrained ImageNet weights frozen** throughout training — no fine-tuning of the vision backbone.

## 🚀 Project Overview

Image captioning is a multimodal deep learning task that combines:

* **Computer Vision** for understanding image content.
* **Natural Language Processing** for generating descriptive text.

This project follows an Encoder-Decoder architecture:

```text
Input Image
     │
     ▼
Pretrained ViT-B/16 Encoder (Frozen)
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

The pretrained ViT encoder splits the input image into patches and converts it into a sequence of visual tokens, without ever updating its weights.

The Transformer-style decoder then generates the caption token by token while attending to:

1. Previous caption tokens using **Masked Self-Attention**.
2. Image features using **Cross-Attention**.

---

## 🧠 Model Architecture

### 1. Pretrained ViT-B/16 Image Encoder (Frozen)

The image encoder is a **pretrained ViT-B/16** (ImageNet weights), used purely as a fixed feature extractor.

```text
Input Image (384 × 384)
    │
    ▼
Patch Embedding (16 × 16 patches)
    │
    ▼
576 Patch Tokens + 1 CLS Token = 577 Tokens
    │
    ▼
Frozen Pretrained ViT-B/16 Transformer Blocks
    │
    ▼
768-Dimensional Visual Tokens
```

Since the model runs at 384×384 instead of the original 224×224 resolution, the pretrained positional embeddings are **bicubically interpolated** from the 224×224 grid (197 tokens) to the 384×384 grid (577 tokens). The original ViT weights are never reinitialized.

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
Cross-Attention ◄──── Visual Tokens from Frozen ViT Encoder
      │
      ▼
Feed Forward Network
      │
      ▼
Vocabulary Predictions
```

---

## ⚙️ Model Configuration

| Parameter               |          Value |
| ------------------------ | -------------: |
| Image Size                |      384 × 384 |
| Batch Size                |               4 |
| Epochs                    |              10 |
| Early Stopping            | None (fixed epochs) |
| Learning Rate              |            3e-4 |
| Model Dimension            |             768 |
| Attention Heads            |              12 |
| Decoder Layers             |               6 |
| Dropout                    |             0.1 |
| Tokenizer                  |             BPE |
| Vocabulary                 | Trained on 100% of training captions |
| Validation Images          |             500 |
| Test Images                |             500 |

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

Only the decoder parameters are optimized. The pretrained ViT encoder stays in eval mode and frozen throughout training.

The model is trained using:

* **AdamW Optimizer** (decoder parameters only)
* **CrossEntropyLoss**
* **Cosine Annealing Learning Rate Scheduler**
* **Mixed Precision Training**
* **Fixed number of epochs (no early stopping)**

Padding tokens are ignored during loss calculation.

---

## 📈 Evaluation Metrics

The final model is evaluated using the following image captioning metrics:

### BLEU-4

Measures the similarity between generated captions and reference captions using n-gram precision.

### ROUGE-L

Measures the longest common subsequence between generated and reference captions, rewarding sentence-level overlap.

### CIDEr

Measures the similarity between generated captions and multiple human-written reference captions using TF-IDF weighted n-gram similarity.

The independent test set contains **500 images** and is used only for final evaluation. All three metrics are implemented from scratch inside the notebook.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Pillow
* tokenizers
* tqdm

---

## 📂 Project Structure

```text
vit-image-captioning/
│
├── vit_image_captioning.ipynb
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
jupyter notebook vit_image_captioning.ipynb
```

The notebook will:

1. Detect the available computing device.
2. Discover and load the image-caption dataset.
3. Train a BPE tokenizer on the training captions.
4. Split the dataset into training, validation, and test sets by unique image.
5. Build the frozen pretrained ViT-B/16 Encoder and Transformer Decoder.
6. Train the decoder only, with the encoder kept frozen.
7. Select the best model using validation performance.
8. Generate captions for test images.
9. Evaluate the model using BLEU-4, ROUGE-L, and CIDEr.

> **Note:** The notebook is set up to run on **Kaggle** by default (paths under `/kaggle/input` and `/kaggle/working`). Adjust these paths if running locally or on another platform.

---

## 🔥 Key Features

* Frozen pretrained ViT-B/16 vision encoder (no backbone fine-tuning).
* Interpolated pretrained positional embeddings for higher-resolution (384×384) input.
* Transformer-style decoder with Cross-Attention.
* Multi-head Self-Attention.
* Causal masking for autoregressive caption generation.
* BPE tokenizer trained on the full training caption corpus.
* Mixed precision training.
* Cosine learning rate scheduling.
* Independent validation and test sets.
* Evaluation using BLEU-4, ROUGE-L, and CIDEr — all implemented from scratch.

---

## 🎯 Future Improvements

Possible improvements include:

* Fine-tuning the ViT encoder (partial or full unfreezing).
* Beam Search for better caption generation.
* Larger and more diverse datasets.
* Larger BPE vocabulary sizes.
* Training for more epochs.
* Using pretrained language models as decoders.
* Applying data augmentation.
* Experimenting with different encoder and decoder architectures.

⭐ If you found this project useful, feel free to star the repository!

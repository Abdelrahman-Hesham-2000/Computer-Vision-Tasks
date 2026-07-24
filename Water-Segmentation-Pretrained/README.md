# 🌊 Water Segmentation Using Multispectral Satellite Images

## 📌 Project Overview

This project focuses on **semantic segmentation of water bodies** from multispectral satellite imagery using deep learning.

Instead of training a single segmentation model, multiple pretrained architectures were implemented, compared, and evaluated. The best-performing model was then selected and fine-tuned to achieve higher segmentation accuracy.

---

# 🎯 Objectives

- Segment water bodies from multispectral satellite images.
- Compare multiple pretrained segmentation models.
- Evaluate each model using standard segmentation metrics.
- Select the best-performing architecture.
- Fine-tune the selected model for improved performance.

---

# 📂 Dataset

The project uses multispectral satellite images together with their corresponding binary segmentation masks.

### Preprocessing

- Reading multispectral GeoTIFF images
- Computing global dataset statistics
- Image normalization
- Data augmentation using Albumentations
- Train/Validation split
- Preparing PyTorch datasets and dataloaders

---

# 🏗️ Models Evaluated

The following segmentation architectures were trained and compared:

- **U-Net + Pretrained ResNet34 Encoder**
- **DeepLabV3+ + Pretrained ResNet50 Encoder**
- **Satellite Pretrained U-Net (TorchGeo)**

All models were trained using the same preprocessing pipeline and evaluation procedure to ensure a fair comparison.

---

# 🔍 Model Comparison

Rather than relying on a single architecture, this project performs a systematic comparison between multiple segmentation models.

Each model was evaluated using identical validation data and the following metrics:

- Dice Score
- Intersection over Union (IoU)
- Precision
- Recall
- F1 Score

Training and validation losses were monitored throughout the training process.

After evaluating all candidate models, the best-performing architecture was selected based on validation performance.

The selected model was then **fine-tuned** by unfreezing the encoder and continuing training to further improve segmentation accuracy.

---

# 📊 Evaluation Metrics

The project uses the following metrics:

- Dice Score
- IoU (Intersection over Union)
- Precision
- Recall
- F1 Score

These metrics provide a comprehensive evaluation of segmentation quality.

---

# 🚀 Workflow

```text
Load Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Data Augmentation
      │
      ▼
Train Multiple Models
      │
      ▼
Compare Performance
      │
      ▼
Select Best Model
      │
      ▼
Fine-Tune Best Model
      │
      ▼
Final Evaluation
      │
      ▼
Prediction Visualization
```

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Segmentation Models PyTorch
- TorchGeo
- Albumentations
- Rasterio
- OpenCV
- NumPy
- Matplotlib

---

# 📁 Project Structure

```text
Water-Segmentation/
│
├── Water_Segmentation_Week2_Final.ipynb
├── README.md
└── Report.pdf
```

---

# 📈 Results

Three different pretrained segmentation architectures were trained and compared.

The comparison demonstrated that pretrained models outperform training from scratch in both convergence speed and segmentation accuracy.

After evaluating all models, the best-performing architecture was selected and fine-tuned to obtain the final segmentation model.

The final model achieved the highest validation performance and produced more accurate water segmentation masks.

---

# 🔮 Future Improvements

- Vision Transformer (ViT) based segmentation
- Larger satellite datasets
- More advanced data augmentation
- Hyperparameter optimization
- Multi-class segmentation
- Real-time inference

Water Segmentation Using Multispectral Satellite Images
Project Overview

This project performs semantic segmentation of water bodies from multispectral satellite imagery using deep learning.

Instead of training a single model, multiple pretrained segmentation architectures are compared. The best-performing model is selected and further fine-tuned to achieve higher segmentation accuracy.

Features
Multispectral satellite image segmentation
Data preprocessing and normalization
Data augmentation using Albumentations
Multiple pretrained segmentation models
Automatic model comparison
Transfer learning
Fine-tuning of the best model
Dice Score and IoU evaluation
Precision, Recall, and F1 metrics
Prediction visualization
Models Evaluated

The following architectures were implemented and compared:

U-Net + ResNet34 (ImageNet pretrained)
DeepLabV3+ + ResNet50
TorchGeo Satellite Pretrained U-Net

All models were trained under identical conditions.

After evaluation, the best-performing model was selected and fine-tuned to produce the final segmentation model.

Technologies Used
Python
PyTorch
Segmentation Models PyTorch
TorchGeo
Albumentations
Rasterio
OpenCV
NumPy
Matplotlib
Project Structure
Water-Segmentation/
│
├── Water_Segmentation_Week2_Final.ipynb
├── README.md
├── README.pdf
├── Report.pdf

Workflow
Load multispectral satellite images.
Preprocess and normalize the dataset.
Apply data augmentation.
Train multiple pretrained segmentation models.
Compare model performance using validation metrics.
Select the best-performing model.
Fine-tune the selected model.
Evaluate the final model.
Visualize segmentation predictions.
Evaluation Metrics

The models were evaluated using:

Dice Score
Intersection over Union (IoU)
Precision
Recall
F1 Score

These metrics were used to compare all candidate models and determine the final architecture.

Results

Multiple segmentation models were evaluated under the same experimental setup.

The best-performing model achieved the highest validation performance and was selected as the final solution. Fine-tuning further improved segmentation accuracy and produced more reliable water segmentation masks.

Future Improvements
Vision Transformer-based segmentation
Larger satellite datasets
More advanced augmentation techniques
Hyperparameter optimization
Multi-class semantic segmentation
Real-time inference

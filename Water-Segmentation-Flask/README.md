# 🌊 Water Segmentation Flask Web Application

A Flask-based web application for **semantic segmentation of water bodies** from satellite images using a **U-Net deep learning model** built with **PyTorch**.

The application allows users to upload a satellite image and generates a segmentation mask highlighting water regions.

---

## 📌 Features

- Upload satellite images through a simple web interface.
- Automatic image preprocessing.
- Water body segmentation using a trained U-Net model.
- Display the predicted segmentation mask.
- Lightweight and easy-to-use Flask application.

---

## 🛠️ Technologies Used

- Python
- Flask
- PyTorch
- OpenCV
- NumPy
- Pillow
- HTML
- CSS
- Jinja2

---

## 📂 Project Structure

```
Water-Segmentation-Flask/
│
├── models/
│   └── .gitkeep
│
├── templates/
│   └── index.html
│
├── uploads/
│   └── .gitkeep
│
├── app.py
├── model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Abdelrahman-Hesham-2000/Computer-Vision-Tasks.git
```

Navigate to the project folder

```bash
cd Computer-Vision-Tasks/Water-Segmentation-Flask
```

Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python app.py
```

Open your browser and navigate to

```
http://127.0.0.1:5000
```

---

## 🧠 Model

The trained U-Net model is **not included** in this repository because it exceeds GitHub's file size limit.

After downloading the model, place it inside:

```

## 🔄 Workflow

```
Satellite Image
        │
        ▼
 Image Preprocessing
        │
        ▼
    U-Net Model
        │
        ▼
 Predicted Mask
        │
        ▼
 Flask Web Interface
```

---

## 📦 Requirements

All required packages are listed in

```
requirements.txt
```

Install them using

```bash
pip install -r requirements.txt


---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐.

"""
Flask deployment for the Water Segmentation U-Net model.

The model expects a 12-band multispectral GeoTIFF (same format used in
training: Coastal Aerosol, Blue, Green, Red, NIR, SWIR1, SWIR2, QA, Merit DEM,
Copernicus DEM, ESA World Cover, Water Occurrence). Users upload a .tif image,
the model predicts a binary water mask, and the app returns both the RGB
preview and the predicted mask as base64-encoded PNGs.
"""

import base64
import io
import os

import numpy as np
import rasterio
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

from model import UNet

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_water.pth")
ALLOWED_EXTENSIONS = {"tif", "tiff"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Load model once at startup
# ----------------------------------------------------------------------
model = UNet(in_channels=12, out_channels=1).to(device)
model_loaded = False

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model_loaded = True
    print(f"[INFO] Loaded model weights from {MODEL_PATH}")
else:
    print(
        f"[WARNING] No weights found at {MODEL_PATH}. "
        "Place your trained 'unet_water.pth' file there before predicting."
    )

model.eval()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_and_normalize(image_path: str) -> np.ndarray:
    """Read a 12-band GeoTIFF and normalize each band to [0, 1]."""
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)

    for i in range(image.shape[0]):
        band = image[i]
        image[i] = (band - band.min()) / (band.max() - band.min() + 1e-8)

    return image


def array_to_base64_png(arr: np.ndarray) -> str:
    """Convert a uint8 HxW or HxWx3 numpy array to a base64 PNG data URI."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_rgb_preview(image: np.ndarray) -> np.ndarray:
    """Build an RGB preview (Red=band3, Green=band2, Blue=band1) as uint8."""
    rgb = np.stack([image[3], image[2], image[1]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return (rgb * 255).astype(np.uint8)


def predict_mask(image: np.ndarray) -> np.ndarray:
    """Run the model on a normalized (C, H, W) image and return a binary mask."""
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()

    return output.squeeze().cpu().numpy()


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_loaded=model_loaded)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model_loaded, "device": str(device)})


@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model weights not loaded on the server."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .tif / .tiff files are supported."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        image = read_and_normalize(save_path)

        if image.shape[0] != 12:
            return jsonify(
                {"error": f"Expected 12 bands, got {image.shape[0]}. Check the input file."}
            ), 400

        mask = predict_mask(image)
        rgb_preview = build_rgb_preview(image)

        mask_uint8 = (mask * 255).astype(np.uint8)
        water_pct = float((mask > 0).mean() * 100)

        return jsonify(
            {
                "rgb_preview": array_to_base64_png(rgb_preview),
                "mask": array_to_base64_png(mask_uint8),
                "water_percentage": round(water_pct, 2),
            }
        )

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to process image: {exc}"}), 500

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

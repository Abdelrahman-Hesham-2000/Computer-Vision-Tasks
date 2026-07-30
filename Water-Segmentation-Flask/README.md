# Water Segmentation — Flask Deployment

A Flask application for deploying the U-Net water segmentation model from the notebook.

## Project Structure

```
water_seg_app/
├── app.py              # Flask server + inference logic
├── model.py            # UNet architecture definition (same as in the notebook)
├── requirements.txt
├── models/
│   └── unet_water.pth  # ⚠️ You need to place this here (trained model weights)
├── templates/
│   └── index.html      # Upload and results UI
└── uploads/             # Temporary files during upload
```

## Important First Step: Save the Model Weights

The notebook trains the model but doesn't save it to disk. Before deploying,
add this line to the last training cell in the notebook (right after the
epoch loop):

```python
torch.save(model.state_dict(), "unet_water.pth")
```

Run the notebook, then copy the resulting `unet_water.pth` file into the
`water_seg_app/models/` folder.

## Running Locally

```bash
cd water_seg_app
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt

# Place unet_water.pth inside models/ before running
python app.py
```

The server will run at `http://localhost:5000`.

> Note: `rasterio` sometimes needs `GDAL` installed on the system.
> On Ubuntu: `sudo apt-get install -y gdal-bin libgdal-dev` before `pip install rasterio`.
> If you're using conda, the easiest way is: `conda install -c conda-forge rasterio`.

## Using the Interface

1. Open `http://localhost:5000` in your browser.
2. Upload a 12-band GeoTIFF image (same format as the training data).
3. Click "Run Model" — you'll see the RGB preview, the predicted mask,
   and the percentage of water in the image.

## Direct API (no UI)

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/image.tif"
```

The response returns JSON containing:
- `rgb_preview`: RGB image as base64
- `mask`: predicted water mask as base64
- `water_percentage`: percentage of pixels classified as water

There's also `GET /health` to confirm the server is running and the model is loaded.

## Running in Production

Don't use `app.run(debug=True)` in production. Use gunicorn instead:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 --timeout 120 app:app
```

`-w 2` = number of workers (adjust based on available CPU cores and memory —
the model uses a fair amount of memory).

### Docker (optional)

If you want to containerize the project, here's a simple `Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120", "app:app"]
```

```bash
docker build -t water-seg-app .
docker run -p 8000:8000 -v $(pwd)/models:/app/models water-seg-app
```

## Ideas for Future Improvements

- Add authentication or rate limiting if this will be hosted publicly.
- Store results on S3/cloud storage instead of deleting them immediately, if you need archiving.
- Use `torch.jit.script` or ONNX to speed up inference in production.
- Add a queue (e.g. Celery/RQ) if images are large and processing could take a while.

from __future__ import annotations

import base64
import mimetypes
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

BPD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BPD_ROOT))

from inference.pipeline import default_pipeline
from utils.config import PROJECT_ROOT
from utils.visualization import draw_open_set_detections, save_reconstruction_grid


APP_ROOT = Path(__file__).resolve().parent
RUNNING_ON_VERCEL = bool(os.getenv("VERCEL"))
RUNTIME_ROOT = Path(tempfile.gettempdir()) / "bpd-open-set" if RUNNING_ON_VERCEL else APP_ROOT / "static"
UPLOAD_DIR = RUNTIME_ROOT / "uploads"
RESULT_DIR = RUNTIME_ROOT / "results"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REQUIRED_ARTIFACTS = [
    PROJECT_ROOT / "outputs" / "weights" / "yolov8n_txl_pbc_best.pt",
    PROJECT_ROOT / "outputs" / "weights" / "autoencoder_best.pt",
    PROJECT_ROOT / "outputs" / "weights" / "centroids.pt",
    PROJECT_ROOT / "outputs" / "weights" / "anomaly_calibration.json",
]

for directory in (UPLOAD_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
_pipeline = None


def artifacts_ready() -> bool:
    return all(path.exists() for path in REQUIRED_ARTIFACTS)


def image_response_url(path: Path, static_filename: str) -> str:
    """Return a static URL locally and an inline image from Vercel's ephemeral runtime."""
    if not RUNNING_ON_VERCEL:
        return url_for("static", filename=static_filename)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def get_pipeline():
    global _pipeline
    if not artifacts_ready():
        missing = [path.name for path in REQUIRED_ARTIFACTS if not path.exists()]
        raise RuntimeError(f"The open-set model is not ready. Missing: {', '.join(missing)}")
    if _pipeline is None:
        _pipeline = default_pipeline()
    return _pipeline


@app.get("/")
def index():
    return render_template("index.html", ready=artifacts_ready())


@app.get("/api/status")
def status():
    return jsonify({"ready": artifacts_ready(), "missing": [path.name for path in REQUIRED_ARTIFACTS if not path.exists()]})


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "The image is too large. Upload a PNG or JPEG smaller than 20 MB."}), 413


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Choose a PNG or JPEG smear image first."}), 400
    uploaded = request.files["image"]
    suffix = Path(uploaded.filename or "").suffix.lower()
    if not uploaded.filename or suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only PNG and JPEG images are supported."}), 400
    filename = f"{int(time.time() * 1000)}_{secure_filename(uploaded.filename)}"
    upload_path = UPLOAD_DIR / filename
    uploaded.save(upload_path)
    try:
        pipeline = get_pipeline()
        image, detections = pipeline.analyze_path(upload_path)
        annotated_name = f"annotated_{Path(filename).stem}.png"
        reconstruction_name = f"reconstructions_{Path(filename).stem}.png"
        cv2.imwrite(str(RESULT_DIR / annotated_name), draw_open_set_detections(image, detections))
        save_reconstruction_grid(detections, RESULT_DIR / reconstruction_name)
        rows = []
        for index, item in enumerate(detections, start=1):
            rows.append(
                {
                    "id": index,
                    "label": item["display_label"],
                    "status": item["status"],
                    "confidence": round(item["confidence"], 4),
                    "mse": round(item["mse"], 6),
                    "cosine_distance": round(item["cosine_distance"], 4),
                    "anomaly_score": round(item["anomaly_score"], 4),
                }
            )
        return jsonify(
            {
                "cells_detected": len(rows),
                "abnormal_cells": sum(row["status"] == "Abnormal" for row in rows),
                "threshold": round(float(pipeline.threshold), 4),
                "detections": rows,
                "input_url": image_response_url(upload_path, f"uploads/{filename}"),
                "annotated_url": image_response_url(RESULT_DIR / annotated_name, f"results/{annotated_name}"),
                "reconstruction_url": image_response_url(RESULT_DIR / reconstruction_name, f"results/{reconstruction_name}") if detections else None,
            }
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=False, threaded=False)

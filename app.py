import os
import pickle
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from huggingface_hub import hf_hub_download

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": [
            "https://hammadfakharkhan-hama-covid-predictor.static.hf.space"
        ]
    }
})

HF_REPO = "hammadfakharkhan/HAMA-Covid-Predictor"
HF_REPO_TYPE = "space"

def download_model(filename):
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        filename=filename
    )

    with open(path, "rb") as f:
        return pickle.load(f)


# Load models
try:
    scaler = download_model("scale.pkl")
    pca = download_model("pca.pkl")
    model = download_model("bag.pkl")

    MODEL_READY = True
    print("All models loaded successfully!")

except Exception as e:
    MODEL_READY = False
    print("Model loading error:", e)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "HAMA Covid Predictor API is running"
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if MODEL_READY else "error",
        "model_loaded": MODEL_READY
    })


@app.route("/predict", methods=["POST"])
def predict():

    if not MODEL_READY:
        return jsonify({
            "error": "Models are not loaded"
        }), 503

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    try:
        image = Image.open(
            request.files["image"]
        ).convert("L")

        image = image.resize((64, 64))

        image_array = np.array(
            image,
            dtype=np.float64
        ).reshape(1, -1)

        scaled = scaler.transform(image_array)
        reduced = pca.transform(scaled)

        prediction = model.predict(reduced)[0]

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(reduced)[0]
            confidence = float(np.max(probabilities)) * 100

        return jsonify({
            "prediction": str(prediction),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

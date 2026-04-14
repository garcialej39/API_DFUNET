from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import numpy as np
import base64
import io

from modelo import DFUNet

app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

# Cargar checkpoint
checkpoint = torch.load("dfunet_final.pth", map_location=device)

IMG_SIZE = checkpoint["img_size"]
MEAN = checkpoint["mean"]
STD = checkpoint["std"]
CLASS_TO_IDX = checkpoint["class_to_idx"]
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Crear modelo
model = DFUNet(num_classes=len(CLASS_TO_IDX), dropout=0.5)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


def preprocess_image(image_pil):
    image = image_pil.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_np = np.array(image).astype(np.float32) / 255.0

    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)

    image_np = (image_np - mean) / std
    image_np = np.transpose(image_np, (2, 0, 1))  # HWC -> CHW

    tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)
    return tensor


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensaje": "API DFUNet funcionando"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "image_base64" not in data:
            return jsonify({"error": "Falta el campo image_base64"}), 400

        image_b64 = data["image_base64"]

        # Por si viene como data:image/jpeg;base64,...
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x = preprocess_image(image).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class_raw = IDX_TO_CLASS[pred_idx]
        confidence = float(probs[pred_idx])
        prob_ulcera = float(probs[1])

        # Etiquetas limpias para la app
        if pred_idx == 1:
            pred_label = "ulcera"
        else:
            pred_label = "no_ulcera"

        return jsonify({
            "prediccion": pred_label,
            "confianza": round(confidence, 4),
            "prob_ulcera": round(prob_ulcera, 4),
            "clase_original": pred_class_raw
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
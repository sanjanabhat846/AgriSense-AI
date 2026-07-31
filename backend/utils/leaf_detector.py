import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# ==========================================================
# Load Leaf Detector Model
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "leaf_detector.keras"

leaf_model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

print("\n========================================")
print("Leaf Detector Model Loaded Successfully")
print("========================================")
import os

print("=" * 60)
print("LEAF MODEL PATH:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))
print("Last Modified:", os.path.getmtime(MODEL_PATH))
print("=" * 60)


# ==========================================================
# Predict Leaf
# ==========================================================

def predict_leaf(image_path):
    """
    Returns:
        (True, confidence)  -> Image is a plant leaf
        (False, confidence) -> Image is NOT a plant leaf
    """

    # Load image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    img = np.array(img, dtype=np.float32)

    # MobileNetV2 preprocessing
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = float(leaf_model.predict(img, verbose=0)[0][0])

    print("\n========== LEAF DETECTOR ==========")
    print(f"Raw Model Output : {prediction:.6f}")

    if prediction < 0.5:
        print("Prediction : LEAF")
        return True, (1 - prediction) * 100
    else:
        print("Prediction : NON LEAF")
        return False, prediction * 100
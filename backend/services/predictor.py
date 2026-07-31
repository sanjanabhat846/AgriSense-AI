import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path

from services.disease_info import DISEASE_INFO

# ==========================================================
# Gating Threshold Configuration
#
# Option C: Direct EfficientNetB0 Softmax Confidence & Entropy Gating
# No separate MobileNetV2 leaf detector model is used.
#
# Production Thresholds (Tuned on 200-image benchmark: F1 = 78.43%, Recall = 80%, Spec = 76%):
#   - CONFIDENCE_THRESHOLD = 40.0%
#   - ENTROPY_THRESHOLD    = 1.15 nats
#
# How to Re-Tune Thresholds on Custom Data:
#   1. Run: py -3.11 training/tune_thresholds.py --dir test_realworld
#   2. Or run: py -3.11 training/benchmark_option_c_200.py
#   3. Update CONFIDENCE_THRESHOLD and ENTROPY_THRESHOLD below.
# ==========================================================

CONFIDENCE_THRESHOLD = 40.0   # Minimum top-1 softmax confidence % required (0-100)
ENTROPY_THRESHOLD    = 1.15   # Maximum Shannon entropy allowed in nats (0 to ln(38)=3.637)


# ==========================================================
# Load Disease Model (Single Inference Engine)
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "model" / "best_model.keras"

DATASET_PATH = BASE_DIR / "dataset" / "raw" / "plantvillage dataset" / "color"

model = tf.keras.models.load_model(MODEL_PATH)

print("\nDisease Model Loaded Successfully!")


# ==========================================================
# Read Class Names
# ==========================================================

CLASS_NAMES = sorted([
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])

print(f"{len(CLASS_NAMES)} Classes Loaded")


# ==========================================================
# Shannon Entropy Helper
# H(p) = - sum( p_i * ln(p_i) ) for i in 1..N
# Measure of prediction uncertainty (0.0 = total certainty, 3.637 = max confusion)
# ==========================================================

def calculate_entropy(probabilities):
    probs = np.clip(probabilities, 1e-12, 1.0)
    probs = probs / np.sum(probs)
    return float(-np.sum(probs * np.log(probs)))


# ==========================================================
# Predict Disease with Direct Softmax Confidence & Entropy Gating
# ==========================================================

def predict_disease(img_path):

    # ======================================================
    # Step 1 : Load & Preprocess Image for EfficientNetB0
    # ======================================================

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    # ======================================================
    # Step 2 : Single-Model EfficientNetB0 Inference
    # ======================================================

    predictions = model.predict(img, verbose=0)[0]

    # Calculate Shannon entropy (uncertainty metric)
    entropy = calculate_entropy(predictions)

    # ======================================================
    # Step 3 : Top 3 Predictions & Top-1 Confidence
    # ======================================================

    top3_idx = predictions.argsort()[-3:][::-1]

    top_predictions = []

    for idx in top3_idx:
        label = CLASS_NAMES[idx]
        crop = label.split("___")[0].replace("_", " ")
        disease = label.split("___")[1].replace("_", " ")

        top_predictions.append({
            "crop": crop,
            "disease": disease,
            "confidence": round(float(predictions[idx] * 100), 2)
        })

    best_idx = top3_idx[0]
    predicted_label = CLASS_NAMES[best_idx]
    top1_confidence = float(predictions[best_idx] * 100.0)

    # ======================================================
    # Step 4 : Softmax Confidence & Entropy Gating
    # Rejects OOD non-leaf images without a secondary classifier
    # ======================================================

    is_low_confidence = top1_confidence < CONFIDENCE_THRESHOLD
    is_high_entropy    = entropy > ENTROPY_THRESHOLD

    if is_low_confidence or is_high_entropy:
        reason_parts = []
        if is_low_confidence:
            reason_parts.append(f"Low confidence ({top1_confidence:.1f}% < {CONFIDENCE_THRESHOLD}%)")
        if is_high_entropy:
            reason_parts.append(f"High uncertainty (entropy = {entropy:.2f} > {ENTROPY_THRESHOLD})")
        gating_reason = " and ".join(reason_parts)

        print("\n========== GATING CHECK ==========")
        print(f"Prediction Rejected: {gating_reason}")

        return {
            "success": False,
            "message": "This image does not appear to contain a recognized plant leaf or disease.",
            "confidence": round(top1_confidence, 2),
            "leaf_confidence": round(top1_confidence, 2),  # Backward compatibility for frontend
            "entropy": round(entropy, 4),
            "gating_reason": gating_reason
        }

    # ======================================================
    # Step 5 : Parse Crop & Disease Name
    # ======================================================

    if "___" in predicted_label:
        crop, disease = predicted_label.split("___", 1)
    else:
        crop = "Unknown"
        disease = predicted_label

    crop = crop.replace("_", " ")
    disease = disease.replace("_", " ")

    # ======================================================
    # Step 6 : Retrieve Disease Advisory Information
    # ======================================================

    disease_info = DISEASE_INFO.get(
        predicted_label,
        {
            "description": "No information available.",
            "cause": "Unknown",
            "symptoms": [],
            "treatment": [],
            "prevention": [],
            "severity": "Unknown"
        }
    )

    # ======================================================
    # Step 7 : Final Result Payload (Preserving API Compatibility)
    # ======================================================

    result = {
        "success": True,
        "crop": crop,
        "disease": disease,
        "confidence": round(top1_confidence, 2),
        "leaf_confidence": round(top1_confidence, 2),  # Backward compatibility
        "entropy": round(entropy, 4),
        "top_predictions": top_predictions,
        "description": disease_info["description"],
        "cause": disease_info["cause"],
        "symptoms": disease_info["symptoms"],
        "treatment": disease_info["treatment"],
        "prevention": disease_info["prevention"],
        "severity": disease_info["severity"]
    }

    print("\n========== TOP 3 PREDICTIONS ==========")
    for i, pred in enumerate(top_predictions, start=1):
        print(f"{i}. {pred['crop']} - {pred['disease']} ({pred['confidence']}%)")

    print(f"Entropy: {entropy:.4f} (Threshold: {ENTROPY_THRESHOLD})")

    print("\n========== FINAL RESULT ==========")
    print(result)

    return result
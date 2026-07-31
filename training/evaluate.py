import tensorflow as tf
import numpy as np
from pathlib import Path

# ============================================================
# Paths
# ============================================================

MODEL_PATH = Path("model/best_model.keras")
DATASET_PATH = Path("dataset/raw/plantvillage dataset/color")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# ============================================================
# Load Model
# ============================================================

print("=" * 60)
print("Loading Model...")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

# ============================================================
# Load Validation Dataset
# ============================================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = validation_dataset.class_names

# ============================================================
# Evaluate
# ============================================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(validation_dataset)

print("\n====================================")
print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")
print("====================================")

# ============================================================
# Predict First 10 Images
# ============================================================

print("\nFirst 10 Predictions\n")

images, labels = next(iter(validation_dataset))

predictions = model.predict(images, verbose=0)

for i in range(10):

    true_index = labels[i].numpy()

    pred_index = np.argmax(predictions[i])

    confidence = np.max(predictions[i]) * 100

    print("-" * 50)

    print("Actual     :", class_names[true_index])

    print("Predicted  :", class_names[pred_index])

    print(f"Confidence : {confidence:.2f}%")
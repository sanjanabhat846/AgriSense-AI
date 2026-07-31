import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path

MODEL_PATH = Path("model/best_model.keras")
DATASET_PATH = Path("dataset/raw/plantvillage dataset/color")

model = tf.keras.models.load_model(MODEL_PATH)

class_names = sorted(
    [folder.name for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

# CHANGE THIS TO THE EXACT IMAGE YOU TESTED IN THE WEB APP
IMAGE_PATH = r"C:\Users\sanja\OneDrive\Desktop\AgriSense-AI\dataset\raw\plantvillage dataset\color\Apple___Apple_scab\0a6812de-7416-4ffe-aba9-307599a02c84___FREC_Scab 2973.JPG"

img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img = image.img_to_array(img)

# ----- Test 1 -----
img1 = img / 255.0
pred1 = model.predict(np.expand_dims(img1, axis=0), verbose=0)[0]

# ----- Test 2 -----
img2 = tf.keras.applications.efficientnet.preprocess_input(img.copy())
pred2 = model.predict(np.expand_dims(img2, axis=0), verbose=0)[0]

print("Test 1 (/255):")
print(class_names[np.argmax(pred1)], np.max(pred1))

print()

print("Test 2 (preprocess_input):")
print(class_names[np.argmax(pred2)], np.max(pred2))
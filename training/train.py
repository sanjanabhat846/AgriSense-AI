import tensorflow as tf
from pathlib import Path
import os

# ============================================================
# AGRISENSE AI - CNN TRAINING
# ============================================================

DATASET_PATH = Path("dataset/raw/plantvillage dataset/color")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10

print("=" * 60)
print("🌱 AGRISENSE AI - CNN TRAINING")
print("=" * 60)

# ============================================================
# Load Dataset
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print(f"\n✅ Number of Classes: {num_classes}")

# ============================================================
# Optimize Dataset Pipeline
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.shuffle(1000).prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ============================================================
# Build CNN Model
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # Much smaller than Flatten()
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n📊 MODEL SUMMARY\n")
model.summary()

# ============================================================
# Train
# ============================================================

print("\n🚀 Training Started...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# ============================================================
# Evaluate
# ============================================================

print("\n📈 Evaluating...\n")

loss, accuracy = model.evaluate(validation_dataset)

print(f"\nValidation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")

# ============================================================
# Save Model
# ============================================================

os.makedirs("model", exist_ok=True)

model.save("model/plant_disease_model.keras")

print("\n✅ Model saved successfully!")
print("Location: model/plant_disease_model.keras")
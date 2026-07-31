import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from pathlib import Path

# ============================================================
# AGRISENSE AI - EFFICIENTNETB0
# ============================================================

DATASET_PATH = Path("dataset/raw/plantvillage dataset/color")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

print("=" * 60)
print("🌱 AGRISENSE AI - EFFICIENTNETB0")
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
NUM_CLASSES = len(class_names)

print(f"\nClasses : {NUM_CLASSES}")

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# ============================================================
# EfficientNetB0 Base Model
# ============================================================

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ============================================================
# Build Model
# ============================================================

inputs = tf.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

print("\n📊 MODEL SUMMARY\n")

model.summary()
# ============================================================
# Compile Model
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# Callbacks
# ============================================================

import os

os.makedirs("model", exist_ok=True)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath="model/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )

]

# ============================================================
# Train Model
# ============================================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=callbacks
)

# ============================================================
# Evaluate
# ============================================================

loss, accuracy = model.evaluate(validation_dataset)

print("\n======================================")
print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")
print("======================================")
# =============================================================================
# AgriSense AI -- Leaf Detector Retraining Script
# File    : training/leaf_train.py
# Run from: Project root (AgriSense-AI/)
# Command : py -3.11 training/leaf_train.py
# Requires: TensorFlow >= 2.9, Python 3.11
#
# Strategy: Two-Phase MobileNetV2 Transfer Learning
# ---------------------------------------------------------------------------
# PHASE 1 (epochs 1-10, LR=1e-3):
#   The MobileNetV2 backbone is completely frozen.
#   Only the classifier head (GlobalAvgPool -> Dropout -> Dense) is trained.
#   This "warms up" the head before touching the backbone, preventing large
#   random gradients from corrupting the pretrained ImageNet weights.
#
# PHASE 2 (epochs 11-30, LR=1e-5):
#   The top 30 layers of MobileNetV2 are unfrozen for fine-tuning.
#   These are the final 5 inverted-residual blocks (layers 125-154), which
#   encode high-level semantics ("what the object is"). They need to adapt
#   from recognising lab-style PlantVillage leaves to real-world leaves.
#   Lower layers (edges, textures, shapes) stay frozen -- they are universal
#   and do not need domain adaptation.
#   LR=1e-5 is deliberately tiny to avoid overshooting the pretrained
#   weight values that are already at a good ImageNet minimum.
#
# Domain Shift Problem Being Solved:
#   The original model was trained on PlantVillage (uniform grey background)
#   vs. natural_images (scenes with complex backgrounds). A real phone camera
#   leaf photo has a complex background, so the old model sees it as NonLeaf.
#   This script addresses the shift via:
#     (1) Augmentation that simulates real-world photo conditions
#     (2) Fine-tuning the backbone to adapt its feature representations
#     (3) Class weighting to compensate for the 8:1 Leaf/NonLeaf imbalance
# =============================================================================

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# =============================================================================
# Section 1: Configuration
# All tunable parameters are defined here. No magic numbers elsewhere.
# =============================================================================

# Dataset and model paths (relative to project root)
DATASET_PATH  = "leaf_detector_dataset"   # Contains Leaf/ and NonLeaf/ subdirs
MODEL_PATH    = "models/leaf_detector.keras"  # Loaded by backend/utils/leaf_detector.py

# Image configuration (must match leaf_detector.py inference preprocessing)
IMG_SIZE   = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE = 32           # Reduce to 16 if you run out of RAM
SEED       = 42           # Reproducible train/val split

# Phase 1 -- frozen backbone, train head only
PHASE1_EPOCHS = 10     # Max epochs; EarlyStopping will terminate earlier if converged
PHASE1_LR     = 1e-3   # Standard Adam LR for head training

# Phase 2 -- fine-tune top layers of MobileNetV2
PHASE2_EPOCHS = 20     # Max epochs; EarlyStopping will terminate earlier if converged
PHASE2_LR     = 1e-5   # Very low LR -- must not damage pretrained representations
UNFREEZE_LAST = 30     # Number of MobileNetV2 layers to unfreeze from the top
                        # MobileNetV2 has 155 layers; this unfreezes layers 125-154

# Image file extensions to count for class weight calculation
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# Section 2: Startup Banner
# =============================================================================

print("=" * 70)
print("  AGRISENSE AI -- LEAF DETECTOR RETRAINING")
print("=" * 70)
print(f"  Python      : {sys.version.split()[0]}")
print(f"  TensorFlow  : {tf.__version__}")
print(f"  Dataset     : {DATASET_PATH}")
print(f"  Model output: {MODEL_PATH}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Phase 1     : up to {PHASE1_EPOCHS} epochs, LR = {PHASE1_LR}")
print(f"  Phase 2     : up to {PHASE2_EPOCHS} epochs, LR = {PHASE2_LR}, "
      f"unfreeze top {UNFREEZE_LAST} layers")
print("=" * 70)


# =============================================================================
# Section 3: Dataset Loading
#
# tf.keras.utils.image_dataset_from_directory scans the directory for
# subdirectory names and assigns integer labels alphabetically:
#   "Leaf"    -> 0
#   "NonLeaf" -> 1
# This matches the label mapping in backend/utils/leaf_detector.py:
#   prediction < 0.5  -> class 0 -> LEAF (True)
#   prediction >= 0.5 -> class 1 -> NON LEAF (False)
#
# The function returns float32 tensors with pixel values in [0, 255].
# No normalisation is done here; MobileNetV2 preprocessing is applied
# inside the model graph (Section 5) so it is automatically skipped
# at inference time (the model expects the same raw-pixel input format
# that leaf_detector.py provides after calling preprocess_input externally).
# =============================================================================

print("\n[1/6] Loading datasets from disk ...")

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: '{DATASET_PATH}'\n"
        f"Run this script from the project root (AgriSense-AI/)."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,       # 80% train, 20% validation
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,                   # Same seed ensures no train/val overlap
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,               # Keep validation deterministic
)

class_names = train_ds.class_names

print(f"\n  Label mapping (alphabetical by folder name):")
for idx, name in enumerate(class_names):
    print(f"    {idx} -> {name}")


# =============================================================================
# Section 4: Dynamic Class Weight Calculation
#
# The dataset has approximately 54,305 Leaf images and 6,899 NonLeaf images
# -- an 8:1 imbalance. Without class weighting, binary cross-entropy loss is
# minimised primarily by fitting the majority class, and the model learns to
# predict "Leaf" for almost everything. This makes NonLeaf severely undertrained.
#
# Formula (sklearn "balanced" strategy):
#   weight[c] = total_samples / (n_classes * count[c])
#
# For Leaf    (majority): weight = ~0.56 (downweighted)
# For NonLeaf (minority): weight = ~4.44 (upweighted)
# Net effect: NonLeaf errors cost ~8x more than Leaf errors in the loss function.
#
# This is recalculated at runtime by counting files, so adding real-world
# photos to leaf_detector_dataset/Leaf/ later automatically adjusts the ratio.
#
# Note on flower/fruit in NonLeaf:
#   The NonLeaf folder contains images from the natural_images dataset:
#   airplane, car, cat, dog, flower, fruit, horse, motorbike, person.
#   Flower and fruit images share green texture with leaves and may introduce
#   label noise. They are retained here per project requirements. To exclude
#   them, move files with 'flower_' or 'fruit_' name prefixes out of
#   leaf_detector_dataset/NonLeaf/ before running this script.
# =============================================================================

print("\n[2/6] Computing dynamic class weights ...")

class_counts = {}
for idx, class_name in enumerate(class_names):
    class_path = os.path.join(DATASET_PATH, class_name)
    count = sum(
        1 for f in os.listdir(class_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    class_counts[idx] = count

total_samples = sum(class_counts.values())
n_classes     = len(class_names)

class_weight = {
    cls: total_samples / (n_classes * count)
    for cls, count in class_counts.items()
}

print(f"\n  {'Class':<12}  {'Images':>8}  {'Share':>7}  {'Weight':>8}")
print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*8}")
for cls, count in class_counts.items():
    pct = count / total_samples * 100
    print(f"  {class_names[cls]:<12}  {count:>8,}  {pct:>6.1f}%  {class_weight[cls]:>8.4f}")

minority_cls = min(class_counts, key=class_counts.get)
majority_cls = max(class_counts, key=class_counts.get)
ratio        = class_weight[minority_cls] / class_weight[majority_cls]
print(f"\n  Minority class upweighted {ratio:.1f}x relative to majority class.")


# =============================================================================
# Section 5: Augmentation Pipeline
#
# Augmentation is placed inside the model graph as a Sequential sub-model.
# Keras automatically applies it during model.fit() (training=True) and
# bypasses it during model.predict() and model.evaluate() (training=False).
# No manual training-flag management is needed anywhere.
#
# Augmentation is applied to raw [0,255] float32 images BEFORE preprocess_input.
# Augmenting in natural pixel space produces physically realistic transformations.
# For example, RandomBrightness in [0,255] space correctly simulates a wide
# range of real-world exposure variations.
#
# Layer rationale:
#   RandomFlip("horizontal_and_vertical")
#       Original used horizontal only. Phone photos can be taken in any
#       orientation; vertical flip covers upside-down leaf shots.
#
#   RandomRotation(factor=0.3)
#       ±108 degrees (factor * 360). Covers overhead, tilted, and oblique
#       camera angles. Original used factor=0.1 (±36°), too conservative.
#
#   RandomZoom(height_factor=0.3, width_factor=0.3)
#       Simulates close-up vs. arm's-length shooting distance. A leaf
#       photographed from 10 cm fills the frame; from 60 cm it is small
#       within a larger scene. factor=0.3 covers this range.
#
#   RandomTranslation(height_factor=0.1, width_factor=0.1)
#       Off-center and partially cropped subjects. Not in original script.
#       Handles photos where the leaf is near the frame edge.
#
#   RandomBrightness(factor=0.2)
#       Simulates direct sunlight, partial shade, deep shade, and indoor
#       artificial lighting. Requires TF >= 2.9 (confirmed: TF 2.21).
#
#   RandomContrast(factor=0.3)
#       Simulates overexposure (bright sky background), haze, and harsh
#       shadow gradients common in outdoor mobile photography.
# =============================================================================

print("\n[3/6] Building augmentation pipeline ...")

augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(factor=0.3),
        layers.RandomZoom(height_factor=0.3, width_factor=0.3),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomBrightness(factor=0.2),
        layers.RandomContrast(factor=0.3),
    ],
    name="augmentation",
)

print("  Augmentation layers (active during training only):")
for aug_layer in augmentation.layers:
    print(f"    - {aug_layer.name}")

# Prefetch datasets into memory pipeline for faster training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)


# =============================================================================
# Section 6: Model Construction
#
# Architecture:
#   Input (224, 224, 3)  [raw pixels, float32, range [0, 255]]
#     -> augmentation     [training mode only: random transforms]
#     -> preprocess_input [maps [0,255] to [-1,1] for MobileNetV2]
#     -> MobileNetV2      [feature extractor, weights='imagenet']
#     -> GlobalAveragePooling2D
#     -> Dropout(0.3)     [regularisation to prevent head overfitting]
#     -> Dense(1, sigmoid) [binary output: 0.0=Leaf, 1.0=NonLeaf]
#
# base_model(x, training=False):
#   The `training=False` argument keeps MobileNetV2's BatchNormalization
#   layers in inference mode throughout both Phase 1 AND Phase 2. This is
#   the correct pattern for transfer learning (confirmed by TF documentation).
#   Using training=True would update BN running statistics from small
#   batches (batch size 32), which destabilises fine-tuning. The stored
#   ImageNet statistics are more reliable than any batch estimate.
# =============================================================================

print("\n[4/6] Building model ...")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load MobileNetV2 with ImageNet weights, no classification top
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False   # Frozen during Phase 1

# Build functional model
inputs  = tf.keras.Input(shape=(224, 224, 3), name="image_input")
x       = augmentation(inputs)                                    # Augment (train only)
x       = tf.keras.applications.mobilenet_v2.preprocess_input(x) # Normalize to [-1, 1]
x       = base_model(x, training=False)                           # Extract features
x       = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x       = layers.Dropout(0.3, name="dropout")(x)
outputs = layers.Dense(1, activation="sigmoid", name="output")(x) # Binary sigmoid

model = models.Model(inputs=inputs, outputs=outputs, name="leaf_detector")

print(f"  Model built: {model.name}")
print(f"  MobileNetV2 layers: {len(base_model.layers)} (all frozen for Phase 1)")


# =============================================================================
# Section 7: Phase 1 -- Train Classifier Head (Backbone Frozen)
# =============================================================================

print("\n" + "=" * 70)
print("  PHASE 1 -- Training Classifier Head (MobileNetV2 Frozen)")
print("=" * 70)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        # AUC (Area Under ROC Curve) is a better metric than accuracy for
        # imbalanced datasets. A degenerate model predicting all-Leaf gets
        # 87% accuracy but only 0.5 AUC. Target val_auc > 0.95 after Phase 1.
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()

phase1_callbacks = [

    # Stop training when val_loss stops improving.
    # patience=5: wait 5 epochs before stopping (original was 3).
    # restore_best_weights=True: reload the best epoch automatically.
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    # Save model only when val_loss improves.
    # This file is loaded by backend/utils/leaf_detector.py at Flask startup.
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    # Reduce LR by 50% if val_loss does not improve for 2 epochs.
    # Allows fine-grained descent without a manual LR decay schedule.
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

print(f"\n  Training for up to {PHASE1_EPOCHS} epochs with class_weight={class_weight}\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    callbacks=phase1_callbacks,
    class_weight=class_weight,
)

p1_best_loss = min(history1.history["val_loss"])
p1_best_acc  = max(history1.history["val_accuracy"])
p1_best_auc  = max(history1.history["val_auc"])

print(f"\n  Phase 1 complete:")
print(f"    Best val_loss     : {p1_best_loss:.4f}")
print(f"    Best val_accuracy : {p1_best_acc:.4f}  ({p1_best_acc * 100:.2f}%)")
print(f"    Best val_AUC      : {p1_best_auc:.4f}")

if p1_best_auc < 0.90:
    print("\n  WARNING: val_AUC < 0.90 after Phase 1.")
    print("  Phase 2 may not improve real-world accuracy significantly.")
    print("  Consider increasing PHASE1_EPOCHS or checking the dataset.")


# =============================================================================
# Section 8: Phase 2 -- Fine-Tune Top 30 Layers of MobileNetV2
#
# MobileNetV2 layer groups (155 total):
#   Layers 0-15    : Initial Conv2D + first inverted residual block
#                    Learn: edges, colour gradients, basic textures
#                    Action: KEEP FROZEN (universal low-level features)
#
#   Layers 16-124  : Middle inverted residual blocks
#                    Learn: shapes, object parts, mid-level patterns
#                    Action: KEEP FROZEN (sufficiently general)
#
#   Layers 125-154 : Final 5 inverted residual blocks (~30 layers)
#                    Learn: high-level semantics -- "what the object is"
#                    These currently encode "PlantVillage lab leaf on grey
#                    background". They must adapt to "leaf in any real scene".
#                    Action: UNFREEZE and fine-tune at LR=1e-5
# =============================================================================

print("\n" + "=" * 70)
print(f"  PHASE 2 -- Fine-Tuning Top {UNFREEZE_LAST} Layers of MobileNetV2")
print("=" * 70)

# Unfreeze entire backbone first, then re-freeze lower layers
base_model.trainable = True

total_layers = len(base_model.layers)
freeze_until = total_layers - UNFREEZE_LAST  # Index of first unfrozen layer

print(f"\n  MobileNetV2 total layers : {total_layers}")
print(f"  Frozen layers            : 0 to {freeze_until - 1}")
print(f"  Unfrozen layers          : {freeze_until} to {total_layers - 1}")

for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

# Count trainable parameters after partial unfreeze
trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
frozen_params    = int(np.sum([np.prod(v.shape) for v in model.non_trainable_variables]))
print(f"  Trainable parameters     : {trainable_params:,}")
print(f"  Frozen parameters        : {frozen_params:,}")

# Must recompile after changing layer trainability.
# New Adam optimizer instance required -- reusing the Phase 1 optimizer
# would carry over accumulated momentum, causing incorrect updates.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
    ],
)

phase2_callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    # Overwrites the Phase 1 checkpoint only if Phase 2 achieves a better val_loss.
    # If Phase 2 diverges, the Phase 1 model remains on disk as the best model.
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    # min_lr=1e-8 (tighter than Phase 1's 1e-7) because the base LR is already
    # 1e-5. Halving below 1e-8 provides no meaningful gradient signal.
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-8,
        verbose=1,
    ),
]

print(f"\n  Training for up to {PHASE2_EPOCHS} epochs with class_weight={class_weight}\n")

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=phase2_callbacks,
    class_weight=class_weight,
)

p2_best_loss = min(history2.history["val_loss"])
p2_best_acc  = max(history2.history["val_accuracy"])
p2_best_auc  = max(history2.history["val_auc"])

print(f"\n  Phase 2 complete:")
print(f"    Best val_loss     : {p2_best_loss:.4f}")
print(f"    Best val_accuracy : {p2_best_acc:.4f}  ({p2_best_acc * 100:.2f}%)")
print(f"    Best val_AUC      : {p2_best_auc:.4f}")


# =============================================================================
# Section 9: Final Summary
# =============================================================================

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"\n  Saved model : {MODEL_PATH}")
print()
print(f"  {'':38}  {'val_loss':>8}  {'val_acc':>8}  {'val_AUC':>8}")
print(f"  {'Phase 1 -- frozen backbone':38}  "
      f"{p1_best_loss:>8.4f}  {p1_best_acc:>8.4f}  {p1_best_auc:>8.4f}")
print(f"  {'Phase 2 -- fine-tuned top 30 layers':38}  "
      f"{p2_best_loss:>8.4f}  {p2_best_acc:>8.4f}  {p2_best_auc:>8.4f}")
print()
print("  IMPORTANT:")
print("    val_accuracy and val_AUC are measured on the in-distribution")
print("    validation split (PlantVillage lab images + natural_images).")
print("    These numbers do NOT reflect real-world mobile camera performance.")
print("    High val_accuracy was already 99.98% before -- that metric is")
print("    misleading. val_AUC > 0.95 AND real-world testing are the true")
print("    indicators of improvement.")
print()
print("  NEXT STEP -- evaluate on real mobile camera leaf photos:")
print("    py -3.11 training/evaluate_realworld.py --image_dir <path_to_photos>")
print()
print("  To add real-world leaf data for further improvement, see:")
print("    training/DATASETS.md")

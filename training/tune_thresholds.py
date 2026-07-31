"""
AgriSense AI - Threshold Tuning Script for Direct Disease Classifier Gating

This script evaluates a validation dataset containing real-world leaf photos
(e.g., test_realworld/Leaf) and obvious non-leaf images (e.g., test_realworld/NonLeaf)
against the 38-class EfficientNetB0 disease classifier.

It calculates Top-1 Softmax Confidence (%) and Shannon Entropy (nats) for each
image and suggests optimal values for CONFIDENCE_THRESHOLD and ENTROPY_THRESHOLD
in backend/services/predictor.py.

Usage:
    py -3.11 training/tune_thresholds.py --dir test_realworld

Expected Directory Structure:
    test_realworld/
        Leaf/         <- Real plant leaf photos
        NonLeaf/      <- Non-leaf photos (cars, planes, animals, objects)

How to Tune Thresholds:
    1. Run this script on your target validation set.
    2. Review the Confidence and Entropy distributions for Leaf vs. NonLeaf.
    3. Choose a CONFIDENCE_THRESHOLD (e.g. 70.0%) that sits between the
       min/5th-percentile of real leaves and max/95th-percentile of non-leaves.
    4. Choose an ENTROPY_THRESHOLD (e.g. 1.80 nats) that sits above the 95th-percentile
       of real leaves and below the median of non-leaves.
    5. Update CONFIDENCE_THRESHOLD and ENTROPY_THRESHOLD at the top of backend/services/predictor.py.
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path

# Config
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "model" / "best_model.keras"
DATASET_PATH = BASE_DIR / "dataset" / "raw" / "plantvillage dataset" / "color"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_metrics(model, img_path):
    """
    Evaluates EfficientNetB0 on a single image and computes:
      - Top-1 Confidence (%)
      - Shannon Entropy H(p) = - sum(p * ln(p)) in nats
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    preds = model.predict(img, verbose=0)[0]
    preds = np.clip(preds, 1e-12, 1.0)  # Avoid log(0)
    preds = preds / np.sum(preds)       # Re-normalize

    top1_conf = float(np.max(preds) * 100.0)
    entropy   = float(-np.sum(preds * np.log(preds)))

    return top1_conf, entropy


def evaluate_folder(model, folder_path):
    confidences = []
    entropies   = []
    filenames   = []

    files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    for f in files:
        fpath = os.path.join(folder_path, f)
        conf, ent = compute_metrics(model, fpath)
        confidences.append(conf)
        entropies.append(ent)
        filenames.append(f)

    return filenames, np.array(confidences), np.array(entropies)


def print_stats(name, confs, ents):
    print(f"\n--- {name} Statistics (n={len(confs)}) ---")
    print(f"  Confidence (%) : Mean={np.mean(confs):.2f}%, Std={np.std(confs):.2f}%, "
          f"Min={np.min(confs):.2f}%, Max={np.max(confs):.2f}%")
    print(f"  Entropy (nats) : Mean={np.mean(ents):.4f}, Std={np.std(ents):.4f}, "
          f"Min={np.min(ents):.4f}, Max={np.max(ents):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune Confidence & Entropy Thresholds for Direct Disease Classifier Gating."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="test_realworld",
        help="Path to validation directory with Leaf/ and NonLeaf/ subfolders.",
    )
    args = parser.parse_args()

    val_path = Path(args.dir)
    leaf_path    = val_path / "Leaf"
    nonleaf_path = val_path / "NonLeaf"

    if not leaf_path.is_dir() or not nonleaf_path.is_dir():
        print(f"ERROR: Expected directory structure:\n  {val_path}/Leaf/\n  {val_path}/NonLeaf/")
        sys.exit(1)

    print(f"Loading EfficientNetB0 model: {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    print("\nEvaluating Leaf images ...")
    leaf_names, leaf_confs, leaf_ents = evaluate_folder(model, leaf_path)

    print("Evaluating NonLeaf images ...")
    nonleaf_names, nonleaf_confs, nonleaf_ents = evaluate_folder(model, nonleaf_path)

    print("=" * 70)
    print("  THRESHOLD TUNING ANALYSIS REPORT")
    print("=" * 70)

    print_stats("Leaf Class", leaf_confs, leaf_ents)
    print_stats("NonLeaf Class", nonleaf_confs, nonleaf_ents)

    # Detailed per-image breakdown
    print("\n" + "=" * 70)
    print("Detailed Image Breakdown:")
    print(f"  {'Filename':<40}  {'Ground Truth':<10}  {'Confidence':>10}  {'Entropy':>10}")
    print("  " + "-" * 75)
    for fname, conf, ent in zip(leaf_names, leaf_confs, leaf_ents):
        print(f"  {fname:<40}  {'LEAF':<10}  {conf:>9.2f}%  {ent:>10.4f}")
    for fname, conf, ent in zip(nonleaf_names, nonleaf_confs, nonleaf_ents):
        print(f"  {fname:<40}  {'NON LEAF':<10}  {conf:>9.2f}%  {ent:>10.4f}")

    # Recommended thresholds
    # Confidence threshold: 5th percentile of leaves vs 95th percentile of non-leaves
    rec_conf = np.percentile(leaf_confs, 10) if len(leaf_confs) > 0 else 65.0
    rec_ent  = np.percentile(leaf_ents, 90)  if len(leaf_ents) > 0  else 1.80

    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION FOR backend/services/predictor.py:")
    print("=" * 70)
    print(f"  CONFIDENCE_THRESHOLD = {rec_conf:.2f}  # Required minimum top-1 softmax confidence %")
    print(f"  ENTROPY_THRESHOLD    = {rec_ent:.4f}  # Allowed maximum Shannon entropy in nats")
    print()
    print("Copy these values into backend/services/predictor.py to update your gating parameters.")


if __name__ == "__main__":
    main()

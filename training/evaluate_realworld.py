# =============================================================================
# AgriSense AI -- Real-World Leaf Detector Evaluation Script
# File    : training/evaluate_realworld.py
# Run from: Project root (AgriSense-AI/)
# Requires: TensorFlow >= 2.9, Python 3.11, Pillow
#
# Purpose:
#   Tests the retrained leaf detector against real mobile-camera images.
#   Mirrors the EXACT preprocessing used in backend/utils/leaf_detector.py
#   so results reflect true production inference behaviour.
#
#   The in-distribution validation set (PlantVillage + natural_images) is
#   NOT a reliable measure of real-world performance. This script is.
#
# Usage modes:
#   1. Single image:
#        py -3.11 training/evaluate_realworld.py --image path/to/leaf.jpg
#
#   2. Directory -- all images expected to be Leaf (unlabeled):
#        py -3.11 training/evaluate_realworld.py --image_dir path/to/leaves/
#
#   3. Labeled directory -- full precision/recall/F1 evaluation:
#        py -3.11 training/evaluate_realworld.py --labeled_dir path/to/labeled/
#
#        Expected folder structure:
#            labeled_dir/
#                Leaf/        <- real-world leaf photos from a phone camera
#                NonLeaf/     <- real-world non-leaf photos (optional)
# =============================================================================

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path


# =============================================================================
# Configuration
# Must match backend/utils/leaf_detector.py exactly.
# =============================================================================

MODEL_PATH = "models/leaf_detector.keras"
IMG_SIZE   = (224, 224)

# Decision threshold (matches leaf_detector.py line: "if prediction < 0.5")
# prediction < THRESHOLD  -> LEAF    (True)
# prediction >= THRESHOLD -> NON LEAF (False)
THRESHOLD = 0.5

# Recognised image extensions
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# Core Prediction Function
#
# Preprocessing pipeline (must be identical to backend/utils/leaf_detector.py):
#   1. Open image with PIL, convert to RGB (strips alpha, handles grayscale)
#   2. Resize to (224, 224) -- MobileNetV2 input size
#   3. Cast to float32 -- pixel values in [0, 255]
#   4. Apply mobilenet_v2.preprocess_input -- maps [0, 255] to [-1, 1]
#   5. Add batch dimension: shape (224,224,3) -> (1,224,224,3)
#   6. Run model.predict -- returns sigmoid output in [0, 1]
#   7. Threshold at 0.5: < 0.5 = Leaf, >= 0.5 = NonLeaf
# =============================================================================

def predict_single(model, image_path):
    """
    Run leaf detection on a single image file.

    Args:
        model     : Loaded Keras model (leaf detector)
        image_path: Path to image file (str or Path)

    Returns:
        is_leaf    (bool)  : True if detected as a leaf
        confidence (float) : Confidence percentage [0-100]
        raw_output (float) : Raw sigmoid model output [0.0-1.0]
    """
    # Load and preprocess (mirrors leaf_detector.py exactly)
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    raw_output = float(model.predict(img, verbose=0)[0][0])

    is_leaf    = raw_output < THRESHOLD
    confidence = (1.0 - raw_output) * 100.0 if is_leaf else raw_output * 100.0

    return is_leaf, confidence, raw_output


# =============================================================================
# Utility Functions
# =============================================================================

def collect_images(directory):
    """Return sorted list of image Paths in a directory (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def print_separator(char="=", width=70):
    print(char * width)


def print_section(title, width=70):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def interpret_recall(leaf_recall_pct):
    """
    Print a human-readable assessment based on real-world leaf recall.
    Recall is the correct metric here: of all true leaf photos, how many
    did the model correctly identify as leaves?
    """
    print()
    if leaf_recall_pct >= 90.0:
        print("  Assessment : PRODUCTION READY")
        print("               Model generalises well to real-world photos.")
    elif leaf_recall_pct >= 70.0:
        print("  Assessment : ACCEPTABLE")
        print("               Good improvement. For production-grade accuracy,")
        print("               add 500-1,000 real-world leaf photos to")
        print("               leaf_detector_dataset/Leaf/ and retrain.")
        print("               See training/DATASETS.md for dataset options.")
    elif leaf_recall_pct >= 50.0:
        print("  Assessment : PARTIAL IMPROVEMENT")
        print("               Augmentation + fine-tuning helped, but domain")
        print("               shift partially persists. Real-world leaf images")
        print("               must be added to fully close the gap.")
        print("               See training/DATASETS.md for dataset options.")
    else:
        print("  Assessment : INSUFFICIENT")
        print("               Real-world leaf images must be added to the Leaf")
        print("               training class before meaningful improvement is")
        print("               possible. See training/DATASETS.md.")


# =============================================================================
# Mode 1: Single Image Evaluation
# =============================================================================

def evaluate_single(model, image_path):
    """Test the model on a single image and print a detailed report."""
    if not os.path.isfile(image_path):
        print(f"ERROR: File not found: {image_path}")
        sys.exit(1)

    print_section(f"Single Image Evaluation")
    print(f"\n  Image     : {image_path}")

    is_leaf, confidence, raw = predict_single(model, image_path)
    label = "LEAF" if is_leaf else "NON LEAF"

    print(f"  Raw output: {raw:.6f}  (threshold = {THRESHOLD})")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {confidence:.2f}%")

    print()
    if is_leaf:
        print("  Result: Image PASSED the leaf gate.")
        print("          It will proceed to the disease classifier.")
    else:
        print("  Result: Image REJECTED as non-leaf.")
        print("          It will NOT proceed to the disease classifier.")
        print(f"\n  Debug: raw output = {raw:.6f}")
        print("          Values near 0.5 indicate the model is uncertain.")
        print("          Values near 1.0 indicate strong NonLeaf prediction.")


# =============================================================================
# Mode 2: Unlabeled Directory (all images expected to be Leaf)
# =============================================================================

def evaluate_directory(model, image_dir):
    """
    Test the model on a directory of images, all assumed to be real leaves.
    Computes the leaf detection rate (recall for Leaf class).
    """
    images = collect_images(image_dir)
    if not images:
        print(f"No images found in: {image_dir}")
        print(f"Supported formats: {IMAGE_SUFFIXES}")
        return

    print_section(f"Directory Evaluation  (all images expected: LEAF)")
    print(f"\n  Directory : {image_dir}")
    print(f"  Images    : {len(images)}")
    print()

    col_w = min(45, max(len(p.name) for p in images) + 2)
    header = f"  {'Filename':<{col_w}}  {'Raw':>8}  {'Prediction':<12}  {'Confidence':>10}"
    print(header)
    print("  " + "-" * (col_w + 36))

    leaf_count = 0

    for img_path in images:
        is_leaf, conf, raw = predict_single(model, img_path)
        label = "LEAF" if is_leaf else "NON LEAF  <--"
        if is_leaf:
            leaf_count += 1
        print(f"  {img_path.name:<{col_w}}  {raw:>8.4f}  {label:<12}  {conf:>9.1f}%")

    total             = len(images)
    detection_rate    = leaf_count / total * 100
    missed            = total - leaf_count

    print("  " + "-" * (col_w + 36))
    print(f"\n  Total images    : {total}")
    print(f"  Detected (LEAF) : {leaf_count}  ({detection_rate:.1f}%)")
    print(f"  Missed (wrong)  : {missed}  ({100.0 - detection_rate:.1f}%)")
    print(f"\n  Real-world leaf detection rate: {detection_rate:.1f}%")

    interpret_recall(detection_rate)


# =============================================================================
# Mode 3: Labeled Directory (Leaf/ and NonLeaf/ subdirs)
# Computes full precision, recall, F1, specificity, and confusion matrix.
# =============================================================================

def evaluate_labeled(model, labeled_dir):
    """
    Full evaluation with ground truth labels.
    Expects:  labeled_dir/Leaf/     <- real leaf photos
              labeled_dir/NonLeaf/  <- real non-leaf photos (optional)
    """
    labeled_path = Path(labeled_dir)
    leaf_dir     = labeled_path / "Leaf"
    nonleaf_dir  = labeled_path / "NonLeaf"

    leaf_images    = collect_images(leaf_dir)    if leaf_dir.is_dir()    else []
    nonleaf_images = collect_images(nonleaf_dir) if nonleaf_dir.is_dir() else []

    if not leaf_images and not nonleaf_images:
        print(f"No images found in: {labeled_dir}")
        print("Expected structure:")
        print(f"    {labeled_dir}/")
        print(f"        Leaf/        <- real-world leaf photos")
        print(f"        NonLeaf/     <- real-world non-leaf photos (optional)")
        return

    print_section("Labeled Evaluation  (Precision / Recall / F1)")
    print(f"\n  Directory: {labeled_dir}")
    print(f"  Leaf images   : {len(leaf_images)}")
    print(f"  NonLeaf images: {len(nonleaf_images)}")

    col_w   = 45
    results = []  # List of (ground_truth_is_leaf, predicted_is_leaf)

    # -------------------------------------------------------------------------
    # Evaluate Leaf images
    # -------------------------------------------------------------------------
    if leaf_images:
        print(f"\n  {'--- Leaf Images (Ground Truth: LEAF) ---'}")
        print(f"\n  {'Filename':<{col_w}}  {'Raw':>8}  {'Predicted':<12}  {'Correct':>7}")
        print("  " + "-" * (col_w + 32))

        for img_path in leaf_images:
            is_leaf, conf, raw = predict_single(model, img_path)
            pred    = "LEAF" if is_leaf else "NON LEAF"
            correct = "YES" if is_leaf else "NO  <--"
            results.append((True, is_leaf))
            print(f"  {img_path.name:<{col_w}}  {raw:>8.4f}  {pred:<12}  {correct:>7}")

    # -------------------------------------------------------------------------
    # Evaluate NonLeaf images
    # -------------------------------------------------------------------------
    if nonleaf_images:
        print(f"\n  {'--- NonLeaf Images (Ground Truth: NON LEAF) ---'}")
        print(f"\n  {'Filename':<{col_w}}  {'Raw':>8}  {'Predicted':<12}  {'Correct':>7}")
        print("  " + "-" * (col_w + 32))

        for img_path in nonleaf_images:
            is_leaf, conf, raw = predict_single(model, img_path)
            pred    = "LEAF" if is_leaf else "NON LEAF"
            # Correct = correctly identified as NonLeaf (is_leaf should be False)
            correct = "YES" if not is_leaf else "NO  <--"
            results.append((False, is_leaf))
            print(f"  {img_path.name:<{col_w}}  {raw:>8.4f}  {pred:<12}  {correct:>7}")

    # -------------------------------------------------------------------------
    # Compute metrics
    # gt=True (real leaf), pred=True (predicted leaf)
    # -------------------------------------------------------------------------
    TP = sum(1 for gt, pred in results if     gt and     pred)  # Leaf correctly detected
    TN = sum(1 for gt, pred in results if not gt and not pred)  # NonLeaf correctly rejected
    FP = sum(1 for gt, pred in results if not gt and     pred)  # NonLeaf wrongly detected as Leaf
    FN = sum(1 for gt, pred in results if     gt and not pred)  # Leaf wrongly rejected

    total       = len(results)
    accuracy    = (TP + TN) / total * 100      if total > 0        else 0.0
    precision   = TP / (TP + FP)       * 100   if (TP + FP) > 0    else 0.0
    recall      = TP / (TP + FN)       * 100   if (TP + FN) > 0    else 0.0
    specificity = TN / (TN + FP)       * 100   if (TN + FP) > 0    else 0.0
    f1_denom    = precision + recall
    f1          = 2 * precision * recall / f1_denom if f1_denom > 0 else 0.0

    print(f"\n  {'=' * 70}")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':25}  {'Predicted LEAF':>14}  {'Predicted NON-LEAF':>18}")
    print(f"  {'Actual LEAF (true +)':25}  {TP:>14}  {FN:>18}")
    print(f"  {'Actual NON-LEAF (true -)':25}  {FP:>14}  {TN:>18}")

    print(f"\n  Metrics:")
    print(f"    Overall accuracy : {accuracy:.1f}%")
    print(f"    Precision        : {precision:.1f}%"
          "   (of Leaf predictions, how many are truly Leaf)")
    print(f"    Recall           : {recall:.1f}%"
          "   (of true Leaf images, how many were detected)")
    print(f"    Specificity      : {specificity:.1f}%"
          "   (of true NonLeaf images, how many were rejected)")
    print(f"    F1 Score         : {f1:.1f}%")

    # Recall is the key metric: "of real leaf photos, how many pass the gate?"
    interpret_recall(recall)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the AgriSense AI leaf detector on real-world mobile "
            "camera images. Run from the project root (AgriSense-AI/)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  py -3.11 training/evaluate_realworld.py --image my_leaf.jpg\n"
            "  py -3.11 training/evaluate_realworld.py --image_dir photos/leaves/\n"
            "  py -3.11 training/evaluate_realworld.py --labeled_dir test_set/"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        metavar="PATH",
        help="Path to a single image file to test.",
    )
    group.add_argument(
        "--image_dir",
        metavar="DIR",
        help="Directory of real-world leaf images (all treated as ground-truth Leaf).",
    )
    group.add_argument(
        "--labeled_dir",
        metavar="DIR",
        help="Directory with Leaf/ and NonLeaf/ subdirectories for full evaluation.",
    )

    args = parser.parse_args()

    # Startup info
    print_separator()
    print("  AgriSense AI -- Real-World Leaf Detector Evaluation")
    print_separator()
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  TensorFlow : {tf.__version__}")

    # Verify model exists
    if not os.path.isfile(MODEL_PATH):
        print(f"\nERROR: Model not found at '{MODEL_PATH}'")
        print("  Run the training script first:")
        print("    py -3.11 training/leaf_train.py")
        sys.exit(1)

    # Load model
    print(f"\n  Loading model: {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  Model loaded successfully.")

    # Dispatch to selected evaluation mode
    if args.image:
        evaluate_single(model, args.image)
    elif args.image_dir:
        evaluate_directory(model, args.image_dir)
    elif args.labeled_dir:
        evaluate_labeled(model, args.labeled_dir)


if __name__ == "__main__":
    main()

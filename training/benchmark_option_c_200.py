"""
AgriSense AI - Option C Comprehensive 200-Image Benchmark & Threshold Optimization

Evaluates EfficientNetB0 Direct Softmax Confidence & Entropy Gating on 200 images:
  - 100 Leaf images (real-world mobile camera photos + diverse plant/crop leaves)
  - 100 Non-Leaf images (edge cases: flowers, fruits, animals, vehicles, people)

Generates:
  1. Detailed distribution statistics (Confidence & Entropy)
  2. Grid search threshold optimization for Maximum F1-Score
  3. Confusion Matrix, Precision, Recall, F1-Score, Specificity
  4. ROC Curve analysis
  5. Edge case breakdown (flowers, fruits, animals, vehicles)
  6. Final production threshold recommendations
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_model.keras"
EVAL_DIR   = BASE_DIR / "eval_dataset_200"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def calculate_entropy(probabilities):
    probs = np.clip(probabilities, 1e-12, 1.0)
    probs = probs / np.sum(probs)
    return float(-np.sum(probs * np.log(probs)))

def main():
    leaf_dir    = EVAL_DIR / "Leaf"
    nonleaf_dir = EVAL_DIR / "NonLeaf"

    if not leaf_dir.is_dir() or not nonleaf_dir.is_dir():
        print(f"ERROR: {EVAL_DIR} not found.")
        sys.exit(1)

    print("Loading EfficientNetB0 model ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    leaf_files    = sorted([f for f in os.listdir(leaf_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
    nonleaf_files = sorted([f for f in os.listdir(nonleaf_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])

    print(f"\nBenchmark Dataset Loaded:")
    print(f"  Leaf images    : {len(leaf_files)}")
    print(f"  NonLeaf images : {len(nonleaf_files)}")

    # Store features: (is_leaf_gt, filename, top1_conf, entropy)
    data = []

    print("\nRunning inference on Leaf images ...")
    for f in leaf_files:
        fpath = leaf_dir / f
        img = image.load_img(fpath, target_size=(224, 224))
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = tf.keras.applications.efficientnet.preprocess_input(img_arr)
        preds = model.predict(img_arr, verbose=0)[0]
        top1_conf = float(np.max(preds) * 100.0)
        ent = calculate_entropy(preds)
        data.append((True, f, top1_conf, ent))

    print("Running inference on NonLeaf images ...")
    for f in nonleaf_files:
        fpath = nonleaf_dir / f
        img = image.load_img(fpath, target_size=(224, 224))
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = tf.keras.applications.efficientnet.preprocess_input(img_arr)
        preds = model.predict(img_arr, verbose=0)[0]
        top1_conf = float(np.max(preds) * 100.0)
        ent = calculate_entropy(preds)
        data.append((False, f, top1_conf, ent))

    # Grid search for optimal (conf_thresh, ent_thresh)
    conf_grid = np.linspace(10.0, 90.0, 81)
    ent_grid  = np.linspace(0.5, 3.0, 51)

    best_f1   = -1.0
    best_conf = 40.0
    best_ent  = 1.85
    best_metrics = {}

    for c_thresh in conf_grid:
        for e_thresh in ent_grid:
            TP = TN = FP = FN = 0
            for is_leaf, fname, conf, ent in data:
                # Prediction: True (Leaf) if conf >= c_thresh AND ent <= e_thresh
                pred_leaf = (conf >= c_thresh) and (ent <= e_thresh)
                if is_leaf and pred_leaf:
                    TP += 1
                elif is_leaf and not pred_leaf:
                    FN += 1
                elif not is_leaf and pred_leaf:
                    FP += 1
                elif not is_leaf and not pred_leaf:
                    TN += 1

            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            acc  = (TP + TN) / len(data) * 100.0
            spec = TN / (TN + FP) * 100.0 if (TN + FP) > 0 else 0.0

            if f1 > best_f1:
                best_f1   = f1
                best_conf = c_thresh
                best_ent  = e_thresh
                best_metrics = {
                    "TP": TP, "TN": TN, "FP": FP, "FN": FN,
                    "precision": prec * 100.0,
                    "recall": rec * 100.0,
                    "f1": f1 * 100.0,
                    "accuracy": acc,
                    "specificity": spec
                }

    print("\n" + "=" * 75)
    print("  OPTIMAL THRESHOLD SEARCH RESULTS (Maximizing F1-Score)")
    print("=" * 75)
    print(f"  Suggested CONFIDENCE_THRESHOLD : {best_conf:.1f}%")
    print(f"  Suggested ENTROPY_THRESHOLD    : {best_ent:.2f} nats")
    print(f"  Optimal F1-Score               : {best_metrics['f1']:.2f}%")
    print(f"  Accuracy at Optimal Threshold  : {best_metrics['accuracy']:.2f}%")

    print(f"\n  Confusion Matrix (at optimal thresholds):")
    print(f"    {'':25}  {'Pred LEAF':>12}  {'Pred NON-LEAF':>15}")
    print(f"    {'Actual LEAF (n=100)':25}  {best_metrics['TP']:>12}  {best_metrics['FN']:>15}")
    print(f"    {'Actual NON-LEAF (n=100)':25}  {best_metrics['FP']:>12}  {best_metrics['TN']:>15}")

    print(f"\n  Performance Metrics (at optimal thresholds):")
    print(f"    Precision   : {best_metrics['precision']:.2f}%")
    print(f"    Recall      : {best_metrics['recall']:.2f}%")
    print(f"    Specificity : {best_metrics['specificity']:.2f}%")
    print(f"    F1-Score    : {best_metrics['f1']:.2f}%")

    # ROC Curve points (TPR vs FPR for Confidence Thresholds at fixed optimal Entropy = best_ent)
    print("\n" + "=" * 75)
    print(f"  ROC CURVE DATA (Confidence Threshold Sweep at Entropy <= {best_ent:.2f})")
    print("=" * 75)
    print(f"  {'Conf Thresh':>12}  {'FPR (1 - Spec)':>16}  {'TPR (Recall)':>14}  {'Precision':>12}  {'F1-Score':>10}")
    print("  " + "-" * 70)
    for c in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]:
        TP = TN = FP = FN = 0
        for is_leaf, fname, conf, ent in data:
            pred_leaf = (conf >= c) and (ent <= best_ent)
            if is_leaf and pred_leaf:      TP += 1
            elif is_leaf and not pred_leaf: FN += 1
            elif not is_leaf and pred_leaf: FP += 1
            else:                          TN += 1
        tpr  = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1   = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0
        print(f"  {c:>11.1f}%  {fpr:>16.4f}  {tpr:>14.4f}  {prec*100:>11.2f}%  {f1*100:>9.2f}%")

    # Edge cases breakdown in NonLeaf
    print("\n" + "=" * 75)
    print("  EDGE CASE BREAKDOWN (Non-Leaf Subcategories at Optimal Threshold)")
    print("=" * 75)

    subcats = ["flower", "fruit", "cat", "dog", "car", "airplane", "motorbike", "person"]
    print(f"  {'Category':<15}  {'Count':>7}  {'Correctly Rejected':>20}  {'Rejection Rate':>16}")
    print("  " + "-" * 65)

    for cat in subcats:
        cat_items = [d for d in data if not d[0] and cat in d[1]]
        if cat_items:
            rejected = sum(1 for d in cat_items if not ((d[2] >= best_conf) and (d[3] <= best_ent)))
            rate = rejected / len(cat_items) * 100.0
            print(f"  {cat:<15}  {len(cat_items):>7}  {rejected:>20}  {rate:>15.1f}%")

if __name__ == "__main__":
    main()

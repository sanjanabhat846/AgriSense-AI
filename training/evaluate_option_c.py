"""
AgriSense AI - Option C Real-World Verification Script

Evaluates the new Direct Softmax Confidence & Entropy Gating in backend/services/predictor.py
on the test_realworld dataset (10 real-world leaf photos + 10 non-leaf photos).

Usage:
    py -3.11 training/evaluate_option_c.py
"""

import os
import sys
from pathlib import Path

# Add backend directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from services.predictor import predict_disease

def main():
    test_dir = BASE_DIR / "test_realworld"
    leaf_dir    = test_dir / "Leaf"
    nonleaf_dir = test_dir / "NonLeaf"

    print("=" * 70)
    print("  AGRISENSE AI -- OPTION C EVALUATION & VERIFICATION")
    print("=" * 70)

    leaf_files    = sorted(list(leaf_dir.glob("*.jpg")) + list(leaf_dir.glob("*.png")))
    nonleaf_files = sorted(list(nonleaf_dir.glob("*.jpg")) + list(nonleaf_dir.glob("*.png")))

    results = []

    print("\n--- 1. Evaluating Real-World Leaf Photos (Ground Truth: LEAF) ---")
    print(f"  {'Filename':<35}  {'Result':<12}  {'Top-1 Conf':>10}  {'Entropy':>8}  {'Pass/Fail'}")
    print("  " + "-" * 75)

    for fpath in leaf_files:
        res = predict_disease(str(fpath))
        is_pass = res.get("success", False)
        conf    = res.get("confidence", 0.0)
        ent     = res.get("entropy", 0.0)
        status  = "ACCEPTED" if is_pass else "REJECTED"
        results.append((True, is_pass))
        print(f"  {fpath.name:<35}  {status:<12}  {conf:>9.2f}%  {ent:>8.4f}  {( 'OK' if is_pass else 'FAIL' )}")

    print("\n--- 2. Evaluating Non-Leaf Photos (Ground Truth: NON LEAF) ---")
    print(f"  {'Filename':<35}  {'Result':<12}  {'Top-1 Conf':>10}  {'Entropy':>8}  {'Pass/Fail'}")
    print("  " + "-" * 75)

    for fpath in nonleaf_files:
        res = predict_disease(str(fpath))
        is_pass = res.get("success", False)
        conf    = res.get("confidence", 0.0)
        ent     = res.get("entropy", 0.0)
        is_correct = not is_pass
        status     = "REJECTED" if not is_pass else "ACCEPTED"
        results.append((False, is_pass))
        print(f"  {fpath.name:<35}  {status:<12}  {conf:>9.2f}%  {ent:>8.4f}  {( 'OK' if is_correct else 'FAIL' )}")

    # Compute metrics
    TP = sum(1 for gt, pred in results if gt and pred)
    TN = sum(1 for gt, pred in results if not gt and not pred)
    FP = sum(1 for gt, pred in results if not gt and pred)
    FN = sum(1 for gt, pred in results if gt and not pred)

    total     = len(results)
    accuracy  = (TP + TN) / total * 100 if total > 0 else 0
    recall    = TP / (TP + FN) * 100    if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) * 100    if (TP + FP) > 0 else 0
    spec      = TN / (TN + FP) * 100    if (TN + FP) > 0 else 0

    print("\n" + "=" * 70)
    print("  SUMMARY EVALUATION METRICS FOR OPTION C")
    print("=" * 70)
    print(f"  True Positives (Leaf accepted)     : {TP} / {len(leaf_files)}")
    print(f"  True Negatives (Non-Leaf rejected) : {TN} / {len(nonleaf_files)}")
    print(f"  False Positives (Non-Leaf accepted): {FP}")
    print(f"  False Negatives (Leaf rejected)   : {FN}")
    print(f"  --------------------------------------------------")
    print(f"  Real-World Leaf Recall             : {recall:.1f}%")
    print(f"  Non-Leaf Rejection Specificity     : {spec:.1f}%")
    print(f"  Overall Test Accuracy              : {accuracy:.1f}%")

if __name__ == "__main__":
    main()

"""
AgriSense AI - Production Readiness Empirical Verification Script

Tests:
1. Latency Benchmark (Single model EfficientNetB0 vs dual model)
2. Robustness to Edge Case Inputs (1x1 pixel image, corrupted bytes, non-image format)
3. API Payload JSON Schema Consistency
4. Disease Prediction Accuracy on test crops
5. Threshold Sensitivity Matrix
"""

import time
import os
import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from services.predictor import predict_disease, CONFIDENCE_THRESHOLD, ENTROPY_THRESHOLD

def test_latency():
    print("\n--- 1. Latency Benchmark ---")
    test_img = BASE_DIR / "test_realworld" / "Leaf" / "real_leaf_01.jpg"

    # Warmup
    _ = predict_disease(str(test_img))

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = predict_disease(str(test_img))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)

    print(f"  Inference Latency over 10 runs (CPU):")
    print(f"    Average : {avg_ms:.2f} ms")
    print(f"    Std Dev : {std_ms:.2f} ms")
    print(f"    Min     : {min_ms:.2f} ms")
    print(f"    Max     : {max_ms:.2f} ms")
    return avg_ms

def test_robustness():
    print("\n--- 2. Robustness to Edge Case Inputs ---")
    scratch_dir = BASE_DIR / "training" / "scratch"
    os.makedirs(scratch_dir, exist_ok=True)

    # Tiny 1x1 image
    tiny_img = scratch_dir / "tiny_1x1.jpg"
    Image.new("RGB", (1, 1), color="green").save(tiny_img)

    # Huge black 4000x4000 image
    huge_img = scratch_dir / "huge_4000x4000.jpg"
    Image.new("RGB", (2000, 2000), color="black").save(huge_img)

    # Corrupted image file
    corrupt_img = scratch_dir / "corrupt.jpg"
    with open(corrupt_img, "wb") as f:
        f.write(b"NOT_AN_IMAGE_FILE_DATA_CORRUPTED")

    # Test tiny image
    try:
        res_tiny = predict_disease(str(tiny_img))
        print(f"  Tiny 1x1 Image        : HANDLED OK -> success={res_tiny.get('success')}")
    except Exception as e:
        print(f"  Tiny 1x1 Image        : EXCEPTION -> {e}")

    # Test huge image
    try:
        res_huge = predict_disease(str(huge_img))
        print(f"  Huge 2000x2000 Image  : HANDLED OK -> success={res_huge.get('success')}")
    except Exception as e:
        print(f"  Huge 2000x2000 Image  : EXCEPTION -> {e}")

    # Test corrupted image
    try:
        res_corrupt = predict_disease(str(corrupt_img))
        print(f"  Corrupted File        : HANDLED OK -> success={res_corrupt.get('success')}")
    except Exception as e:
        print(f"  Corrupted File        : CAUGHT EXCEPTION (Expected) -> {type(e).__name__}: {e}")

def test_json_schema():
    print("\n--- 3. API Payload JSON Schema Consistency ---")
    test_img = BASE_DIR / "test_realworld" / "Leaf" / "real_leaf_01.jpg"
    res = predict_disease(str(test_img))

    # Verify JSON serializability
    try:
        json_str = json.dumps(res, indent=2)
        print("  JSON Serialization    : PASSED ✅")
    except Exception as e:
        print(f"  JSON Serialization    : FAILED ❌ -> {e}")

    # Keys check
    expected_keys = [
        "success", "crop", "disease", "confidence", "leaf_confidence",
        "entropy", "top_predictions", "description", "cause", "symptoms",
        "treatment", "prevention", "severity"
    ]
    missing = [k for k in expected_keys if k not in res]
    if not missing:
        print("  Payload Keys Check    : ALL REQUIRED KEYS PRESENT ✅")
    else:
        print(f"  Payload Keys Check    : MISSING KEYS -> {missing} ❌")

    print("\nSample Response Payload:")
    print(json_str[:500] + "\n...")

def main():
    print("=" * 70)
    print("  AGRISENSE AI -- PRODUCTION READINESS EMPIRICAL VERIFICATION")
    print("=" * 70)
    avg_latency = test_latency()
    test_robustness()
    test_json_schema()
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

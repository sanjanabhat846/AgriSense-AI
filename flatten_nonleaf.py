import os
import shutil
from pathlib import Path

SOURCE = Path("leaf_detector_dataset/NonLeaf/natural_images")
DEST = Path("leaf_detector_dataset/NonLeaf")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

count = 0

for root, dirs, files in os.walk(SOURCE):
    for file in files:

        ext = Path(file).suffix.lower()

        if ext in image_extensions:

            src = Path(root) / file

            new_name = f"{count}_{file}"

            dst = DEST / new_name

            shutil.copy2(src, dst)

            count += 1

print(f"Copied {count} images successfully!")
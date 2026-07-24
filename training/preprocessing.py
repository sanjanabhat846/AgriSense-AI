from pathlib import Path
import cv2

# Dataset paths
RAW_DATASET = Path("dataset/raw/plantvillage dataset/color")
OUTPUT_DATASET = Path("dataset/processed")

# Create output folder
OUTPUT_DATASET.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🌱 AGRISENSE AI - IMAGE PREPROCESSING")
print("=" * 60)

print("Reading images...\n")

processed_count = 0

for class_folder in RAW_DATASET.iterdir():

    if not class_folder.is_dir():
        continue

    output_class = OUTPUT_DATASET / class_folder.name
    output_class.mkdir(exist_ok=True)

    for image_path in class_folder.glob("*"):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, (224, 224))

        output_path = output_class / image_path.name

        cv2.imwrite(str(output_path), image)

        processed_count += 1

print(f"✅ Images Processed : {processed_count}")
print("\nFinished Successfully 🚀")
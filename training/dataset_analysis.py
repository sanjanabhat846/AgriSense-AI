from pathlib import Path
from PIL import Image

dataset_path = Path("dataset/raw/plantvillage dataset/color")

# Find the first image in the dataset
sample_image = None

for folder in dataset_path.iterdir():
    if folder.is_dir():
        images = list(folder.glob("*"))
        if images:
            sample_image = images[0]
            break

# Open the image
img = Image.open(sample_image)

print("=" * 60)
print("🌱 SAMPLE IMAGE INFORMATION")
print("=" * 60)

print(f"Image Path   : {sample_image}")
print(f"Image Size   : {img.size}")
print(f"Image Mode   : {img.mode}")
print(f"Image Format : {img.format}")
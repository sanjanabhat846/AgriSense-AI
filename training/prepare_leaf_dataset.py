import os
import shutil

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
SOURCE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "plantvillage dataset",
    "color"
)
DEST_DIR = os.path.join(BASE_DIR, "leaf_detector_dataset", "Leaf")

os.makedirs(DEST_DIR, exist_ok=True)

count = 0

# Valid image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

print("Copying leaf images...\n")
print("SOURCE:", SOURCE_DIR)
print("Exists:", os.path.exists(SOURCE_DIR))
print("Total folders:", len(os.listdir(SOURCE_DIR)))

# Traverse all disease folders
for folder in os.listdir(SOURCE_DIR):

    folder_path = os.path.join(SOURCE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    for image_name in os.listdir(folder_path):

        if image_name.lower().endswith(IMAGE_EXTENSIONS):

            source = os.path.join(folder_path, image_name)

            # Rename image to avoid duplicate filenames
            new_name = f"{folder}_{image_name}"

            destination = os.path.join(DEST_DIR, new_name)

            shutil.copy2(source, destination)

            count += 1

print("\n===================================")
print(f"Total Leaf Images Copied : {count}")
print("Destination :", DEST_DIR)
print("===================================")
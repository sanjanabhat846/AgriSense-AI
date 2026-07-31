# Real-World Leaf Datasets for AgriSense AI
### training/DATASETS.md

This document lists publicly available real-world leaf datasets that can be
merged into `leaf_detector_dataset/Leaf/` to fix the remaining domain shift
after the augmentation + fine-tuning retraining (Option C, Phase 1).

---

## Why Real-World Data Is Needed

The current training dataset consists of:
- **Leaf class** : 54,305 PlantVillage images — uniform grey background, controlled lighting
- **NonLeaf class**: ~6,899 natural_images photos — airplanes, cars, cats, dogs, etc.

The improved training script (Phase 1 of Option C) applies augmentation and
MobileNetV2 fine-tuning to partially close this gap. Expected improvement:
**~55–70% real-world leaf detection** vs. ~5–20% with the old model.

For production-grade accuracy (90%+), 500–1,000 real-world leaf photos must
be added to the Leaf class. These photos directly sample the target domain
(phone camera images with complex backgrounds).

---

## Dataset 1 — PlantDoc ⭐ RECOMMENDED FIRST CHOICE

**Why it helps**: Real-world diseased plant images photographed in fields and
gardens with complex backgrounds, variable lighting, and multiple leaves
per frame. Directly overlaps with PlantVillage disease classes in AgriSense AI.

| Property | Detail |
|---|---|
| Source | https://github.com/pratikkayal/PlantDoc-Dataset |
| Size | 2,569 images across 27 disease classes |
| Conditions | Outdoor field photographs — NOT lab-controlled |
| License | Creative Commons (research/non-commercial use) |

**How to integrate:**
```bash
# Clone the repository
git clone https://github.com/pratikkayal/PlantDoc-Dataset

# Copy all images from all class folders into the Leaf training class
# (Windows PowerShell example):
Get-ChildItem -Path "PlantDoc-Dataset" -Recurse -Include "*.jpg","*.png" |
    Copy-Item -Destination "leaf_detector_dataset\Leaf\"

# Retrain
py -3.11 training/leaf_train.py
```

---

## Dataset 2 — iNaturalist (Plants)

**Why it helps**: Millions of real-world plant photos submitted by citizen
scientists using smartphone cameras worldwide. This is the closest dataset
to AgriSense AI's actual use case. Diverse backgrounds, lighting conditions,
and geographic locations.

| Property | Detail |
|---|---|
| Source (web) | https://www.inaturalist.org/observations |
| Kaggle | https://www.kaggle.com/c/inaturalist-2021 |
| Filter | Taxon = Plantae, has_photo = true, quality_grade = research |
| Size | Hundreds of thousands of plant photos |
| License | CC BY-NC (non-commercial research use) |

**How to integrate (via iNaturalist API):**
```python
# Download ~500 real-world plant leaf images via iNaturalist API
import requests, os, urllib.request

url = (
    "https://api.inaturalist.org/v1/observations"
    "?taxon_name=Plantae&has[]=photos&quality_grade=research&per_page=200&page=1"
)
response = requests.get(url).json()

os.makedirs("leaf_detector_dataset/Leaf/inat", exist_ok=True)
for obs in response["results"]:
    if obs.get("photos"):
        photo_url = obs["photos"][0]["url"].replace("square", "medium")
        filename  = f"inat_{obs['id']}.jpg"
        filepath  = f"leaf_detector_dataset/Leaf/inat/{filename}"
        try:
            urllib.request.urlretrieve(photo_url, filepath)
        except Exception:
            pass
```

---

## Dataset 3 — LeafSnap (Field Collection)

**Why it helps**: Designed specifically for leaf identification. Includes
both a lab version (white background) and a field version (natural background).
**Use only the field version** — the lab version has the same problem as
PlantVillage and will not help with domain shift.

| Property | Detail |
|---|---|
| Source | http://leafsnap.com/dataset/ |
| Size | ~23,000+ field leaf images across 185 North American tree species |
| License | Free for research use |

**How to integrate:**
```bash
# After downloading the dataset:
# Copy images from the 'field/' subdirectories only (NOT 'lab/')
Get-ChildItem -Path "leafsnap-dataset\dataset\images\field" -Recurse -Include "*.jpg" |
    Copy-Item -Destination "leaf_detector_dataset\Leaf\"
```

---

## Dataset 4 — Kaggle Plant Pathology 2020

**Why it helps**: High-quality real-world apple leaf photos taken in orchards.
Natural backgrounds (grass, sky, overlapping leaves). Manageable download size.

| Property | Detail |
|---|---|
| Source | https://www.kaggle.com/c/plant-pathology-2020-fgvc7 |
| Size | 3,642 images |
| License | Kaggle competition (research use) |

**How to integrate:**
```bash
# Requires Kaggle API (pip install kaggle)
kaggle competitions download -c plant-pathology-2020-fgvc7
Expand-Archive plant-pathology-2020-fgvc7.zip -DestinationPath plant_path_2020

# Copy training images into the Leaf class
Copy-Item -Path "plant_path_2020\train\*.jpg" -Destination "leaf_detector_dataset\Leaf\"
```

---

## How to Merge Any Dataset

```
Step 1: Download images from any source above.

Step 2: Copy them directly into leaf_detector_dataset/Leaf/
        (No renaming needed. Any filename works.)

Step 3: Retrain -- class weights recalculate automatically:
        py -3.11 training/leaf_train.py

Step 4: Evaluate real-world accuracy:
        py -3.11 training/evaluate_realworld.py --labeled_dir my_test_set/
```

> **Note**: Do NOT add field/real-world images to the NonLeaf class.
> The existing natural_images categories (airplane, car, cat, dog, horse,
> motorbike, person) are sufficient and genuinely distinct from leaves.

---

## Expected Improvement vs. Real Images Added

| Real-world leaf images added | Expected leaf detection rate |
|---|---|
| 0 (augmentation + fine-tuning only) | ~55–70% |
| 200–500 images | ~75–85% |
| 500–1,000 images | ~88–93% |
| 1,000–3,000 images | ~92–96% (production-grade) |

**Diversity matters more than quantity.**
500 images across 5 different backgrounds and 3 lighting conditions
outperforms 1,000 images all taken in the same location.

For minimum viable production coverage, aim for:
- 5+ different background types (grass, soil, concrete, indoors, mixed foliage)
- 3+ lighting conditions (bright sun, shade, overcast/indoor)
- 3+ distances (close-up, arm's length, partial view)

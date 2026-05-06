import json
import os
from collections import defaultdict
import shutil
from constants import ANNOTATIONS_PATH, RAW_IMAGES_DIR, YOLO_IMAGES_DIR, YOLO_LABELS_DIR

os.makedirs(YOLO_IMAGES_DIR, exist_ok=True)
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# Hold vs. Wall
YOLO_CLASS_ID = 0

with open(ANNOTATIONS_PATH, "r") as f:
    annotations = json.load(f)

by_image = defaultdict(list)
for ann in annotations:
    by_image[ann["filename"]].append(ann)

for filename, anns in by_image.items():
    image_src = os.path.join(RAW_IMAGES_DIR, filename)
    image_dst = os.path.join(YOLO_IMAGES_DIR, filename)

    if not os.path.exists(image_src):
        print(f"Warning: missing image {image_src}")
        continue

    # Copy image into YOLO images folder
    shutil.copy2(image_src, image_dst)

    label_filename = f"{os.path.splitext(filename)[0]}.txt"
    label_path = os.path.join(YOLO_LABELS_DIR, label_filename)

    with open(label_path, "w") as out:
        for ann in anns:
            image_width = ann["width"]
            image_height = ann["height"]

            left = ann["bbox_left"]
            top = ann["bbox_top"]
            width = ann["bbox_width"]
            height = ann["bbox_height"]

            x_center = (left + width / 2) / image_width
            y_center = (top + height / 2) / image_height
            yolo_width = width / image_width
            yolo_height = height / image_height

            out.write(
                f"{YOLO_CLASS_ID} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{yolo_width:.6f} {yolo_height:.6f}\n"
            )

print(f"Converted {len(by_image)} images to YOLO format.")
print(f"Images written to: {YOLO_IMAGES_DIR}")
print(f"Labels written to: {YOLO_LABELS_DIR}")
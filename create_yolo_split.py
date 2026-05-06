# create_yolo_split.py

import json
import os
import shutil
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

from constants import (
    ANNOTATIONS_PATH,
    RAW_IMAGES_DIR,
    SPLIT_MANIFEST_PATH,
    YOLO_IMAGES_DIR,
    YOLO_LABELS_DIR,
)

YOLO_CLASS_ID = 0
RANDOM_STATE = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

def get_representative_label(anns):
    labels = [ann["label_value"] for ann in anns]
    return Counter(labels).most_common(1)[0][0]

def yolo_bbox_from_annotation(ann):
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

    return x_center, y_center, yolo_width, yolo_height


def write_yolo_label_file(label_path, annotations):
    with open(label_path, "w") as out:
        for ann in annotations:
            x_center, y_center, yolo_width, yolo_height = yolo_bbox_from_annotation(ann)

            out.write(
                f"{YOLO_CLASS_ID} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{yolo_width:.6f} {yolo_height:.6f}\n"
            )


def copy_image_and_label(filename, annotations, split):
    image_src = os.path.join(RAW_IMAGES_DIR, filename)
    image_dst_dir = os.path.join(YOLO_IMAGES_DIR, split)
    label_dst_dir = os.path.join(YOLO_LABELS_DIR, split)

    os.makedirs(image_dst_dir, exist_ok=True)
    os.makedirs(label_dst_dir, exist_ok=True)

    if not os.path.exists(image_src):
        print(f"Warning: missing image {image_src}")
        return False

    image_dst = os.path.join(image_dst_dir, filename)
    shutil.copy2(image_src, image_dst)

    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(label_dst_dir, label_filename)

    write_yolo_label_file(label_path, annotations)

    return True


def main():
    with open(ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)

    by_image = defaultdict(list)

    for ann in annotations:
        filename = ann["filename"]

        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        by_image[filename].append(ann)

    image_samples = []

    for filename, anns in by_image.items():
        image_samples.append({
            "filename": filename,
            "stratify_label": get_representative_label(anns),
            "annotations": anns,
        })

    labels = [sample["stratify_label"] for sample in image_samples]

    train_samples, temp_samples = train_test_split(
        image_samples,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    temp_labels = [sample["stratify_label"] for sample in temp_samples]

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_labels
    )

    split_to_samples = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    manifest = {
        "train": [],
        "val": [],
        "test": [],
    }

    for split, samples in split_to_samples.items():
        for sample in samples:
            filename = sample["filename"]
            success = copy_image_and_label(
                filename,
                sample["annotations"],
                split
            )

            if success:
                manifest[split].append(filename)

    os.makedirs(os.path.dirname(SPLIT_MANIFEST_PATH), exist_ok=True)

    with open(SPLIT_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print("YOLO split complete.")
    print(f"Train images: {len(manifest['train'])}")
    print(f"Val images: {len(manifest['val'])}")
    print(f"Test images: {len(manifest['test'])}")
    print(f"Split manifest saved to: {SPLIT_MANIFEST_PATH}")
    print(f"YOLO images saved to: {YOLO_IMAGES_DIR}")
    print(f"YOLO labels saved to: {YOLO_LABELS_DIR}")


if __name__ == "__main__":
    main()
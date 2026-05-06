import os

DATA_DIR = "data"
IMAGES_DIR = "images"
MASKS_DIR = "masks"
LABELS_DIR = "labels"
MODEL_DIR = "models"

RAW_DIR = os.path.join(DATA_DIR, "raw")
CROPPED_DIR = os.path.join(DATA_DIR, "cropped")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
SEGMENTED_DIR = os.path.join(DATA_DIR, "segmented")

SEGMENTED_SNAKES_DIR = os.path.join(SEGMENTED_DIR, "snakes")
SEGMENTED_KMEANS_DIR = os.path.join(SEGMENTED_DIR, "kmeans")
SEGMENTED_ANYTHING_DIR = os.path.join(SEGMENTED_DIR, "anything")
SIFT_FEATURES_DIR = os.path.join(DATA_DIR, "sift_features")

PREPROCESSED_CACHE_DIR = os.path.join(PREPROCESSED_DIR, "preprocessed_cache")
PREPROCESSED_CACHE_IMAGES_DIR = os.path.join(PREPROCESSED_CACHE_DIR, IMAGES_DIR)
PREPROCESSED_CACHE_MANIFEST = os.path.join(PREPROCESSED_CACHE_DIR, "preprocessed.json")
SEGMENTED_CACHE_DIR = os.path.join(PREPROCESSED_DIR, "segmented_cache")
SEGMENTED_CACHE_IMAGES_DIR = os.path.join(SEGMENTED_CACHE_DIR, IMAGES_DIR)
SEGMENTED_CACHE_MANIFEST = os.path.join(SEGMENTED_CACHE_DIR, "segmentation.json")

ANNOTATIONS_PATH = os.path.join(RAW_DIR, "annotations.json")
RAW_IMAGES_DIR = os.path.join(RAW_DIR, IMAGES_DIR)

YOLO_DIR = os.path.join(RAW_DIR, "yolo")
YOLO_DATA_YAML = os.path.join(YOLO_DIR, "holds.yaml")
YOLO_IMAGES_DIR = os.path.join(YOLO_DIR, IMAGES_DIR)
YOLO_LABELS_DIR = os.path.join(YOLO_DIR, LABELS_DIR)
SPLIT_MANIFEST_PATH = os.path.join(DATA_DIR, "splits", "image_split.json")
YOLO_IMAGES_TRAIN_DIR = os.path.join(YOLO_IMAGES_DIR, "train")
YOLO_IMAGES_VAL_DIR = os.path.join(YOLO_IMAGES_DIR, "val")
YOLO_IMAGES_TEST_DIR = os.path.join(YOLO_IMAGES_DIR, "test")
YOLO_LABELS_TRAIN_DIR = os.path.join(YOLO_LABELS_DIR, "train")
YOLO_LABELS_VAL_DIR = os.path.join(YOLO_LABELS_DIR, "val")
YOLO_LABELS_TEST_DIR = os.path.join(YOLO_LABELS_DIR, "test")
YOLO_DETECTOR_PATH = os.path.join(MODEL_DIR, "hold_detector.pt")

SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = os.path.join(MODEL_DIR, "sam_vit_h_4b8939.pth")

CLASSIFICATIONS = {
    0: "Jug",
    1: "Crimp",
    2: "Pinch",
    3: "Sloper",
    4: "Pocket",
    5: "Volume",
    6: "Unknown"
}

def create_directories(dir_path):
    imgs_path = os.path.join(dir_path, IMAGES_DIR)
    masks_path = os.path.join(dir_path, MASKS_DIR)
    os.makedirs(imgs_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)   

    return imgs_path, masks_path
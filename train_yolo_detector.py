import os
import shutil

from ultralytics import YOLO

from constants import (
    YOLO_DATA_YAML,
    MODEL_DIR,
    YOLO_DETECTOR_PATH,
)

BASE_MODEL = "yolo11n.pt"

PROJECT_DIR = os.path.join("runs", "hold_detector")
RUN_NAME = "yolo_hold_detector"

EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8

# Windows-friendly settings
DEVICE = 0      
WORKERS = 0     

def main():
    # os.makedirs(MODEL_DIR, exist_ok=True)

    # if not os.path.exists(YOLO_DATA_YAML):
    #     raise FileNotFoundError(f"Could not find YOLO yaml: {YOLO_DATA_YAML}")

    # model = YOLO(BASE_MODEL)

    # model.train(
    #     data=YOLO_DATA_YAML,
    #     epochs=EPOCHS,
    #     imgsz=IMG_SIZE,
    #     batch=BATCH_SIZE,
    #     project=PROJECT_DIR,
    #     name=RUN_NAME,
    #     patience=10,
    #     device=DEVICE,
    #     workers=WORKERS,
    #     exist_ok=True,
    # )

    best_model_path = os.path.join(
        MODEL_DIR,
        "best.pt",
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Could not find trained model: {best_model_path}")

    trained_model = YOLO(best_model_path)

    print("Validating best model...")
    trained_model.val(
        data=YOLO_DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
    )

    shutil.copy2(best_model_path, YOLO_DETECTOR_PATH)

    print("YOLO training complete.")
    print(f"Best model from run: {best_model_path}")
    print(f"Copied detector model to: {YOLO_DETECTOR_PATH}")


if __name__ == "__main__":
    main()
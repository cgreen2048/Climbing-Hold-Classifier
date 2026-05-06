import os
import json
import cv2
import joblib
import torch
import torch.nn as nn
import argparse
from ultralytics import YOLO
from torchvision import models, transforms
from segment_anything import sam_model_registry, SamPredictor

from constants import (
    MODEL_DIR,
    RAW_IMAGES_DIR,
    ANNOTATIONS_PATH,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
    CLASSIFICATIONS
)

from create_classification_model import (
    clamp_bbox,
    pad_and_clamp_bbox,
    normalize_img_size,
    clahe_lighting_bgr,
    segment_one_image
)

def get_bbox_for_input_img(img_path):
    filename = os.path.basename(img_path)

    with open(ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)

    for ann in annotations:
        if ann["filename"] == filename:
            return (
                int(ann["bbox_left"]),
                int(ann["bbox_top"]),
                int(ann["bbox_width"]),
                int(ann["bbox_height"]),
            )

    raise ValueError(f"No bbox found for image: {filename}")

def load_model_artifacts(device, num_classes=7):
    resnet_path = os.path.join(MODEL_DIR, "model4_resnet.pt")
    scaler_path = os.path.join(MODEL_DIR, "model4_scaler.joblib")
    svm_path = os.path.join(MODEL_DIR, "model4_svm.joblib")

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(resnet_path, map_location=device))
    model.to(device)
    model.eval()

    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    scaler = joblib.load(scaler_path)
    clf = joblib.load(svm_path)

    return feature_extractor, scaler, clf

def preprocess_frame_bbox(
    frame_bgr,
    bbox_left,
    bbox_top,
    bbox_width,
    bbox_height,
    output_size=256,
    pad=4,
):
    raw_h, raw_w = frame_bgr.shape[:2]

    left, top, width, height = clamp_bbox(
        bbox_left,
        bbox_top,
        bbox_width,
        bbox_height,
        raw_w,
        raw_h,
    )

    left, top, width, height = pad_and_clamp_bbox(
        left,
        top,
        width,
        height,
        pad,
        raw_w,
        raw_h,
    )

    cropped = frame_bgr[top:top + height, left:left + width]

    normalized, resize_meta = normalize_img_size(cropped, size=output_size)
    equalized = clahe_lighting_bgr(normalized)

    metadata = {
        "bbox_left": left,
        "bbox_top": top,
        "bbox_width": width,
        "bbox_height": height,
        "raw_w": raw_w,
        "raw_h": raw_h,
        "rotation": 1,
        "label_name": None,
        "label_value": None,
    }

    metadata = metadata | resize_meta

    return equalized, metadata

def detect_best_hold_bbox(detector, frame_bgr, conf_threshold=0.25):
    results = detector(frame_bgr, conf=conf_threshold)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    best_box = max(boxes, key=lambda box: float(box.conf[0]))

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    bbox_left = x1
    bbox_top = y1
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    return bbox_left, bbox_top, bbox_width, bbox_height

def classify_frame_with_bbox(
    frame_bgr,
    bbox_left,
    bbox_top,
    bbox_width,
    bbox_height,
    sam_predictor,
    feature_extractor,
    scaler,
    svm_clf,
    device,
):
    preprocessed_img, metadata = preprocess_frame_bbox(
        frame_bgr,
        bbox_left,
        bbox_top,
        bbox_width,
        bbox_height,
    )

    segmented_img, _ = segment_one_image(
        preprocessed_img,
        sam_predictor,
        metadata["new_h"],
        metadata["new_w"],
        metadata["y_offset"],
        metadata["x_offset"],
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = features.flatten(start_dim=1)
        features_np = features.cpu().numpy()

    features_scaled = scaler.transform(features_np)
    pred = svm_clf.predict(features_scaled)[0]

    return {
        "prediction": int(pred),
        "bbox": {
            "left": bbox_left,
            "top": bbox_top,
            "width": bbox_width,
            "height": bbox_height,
        },
        "segmented_img": segmented_img,
    }


def classify_frame_with_yolo(
    frame_bgr,
    detector,
    sam_predictor,
    feature_extractor,
    scaler,
    svm_clf,
    device,
):
    bbox = detect_best_hold_bbox(detector, frame_bgr)

    if bbox is None:
        return None

    bbox_left, bbox_top, bbox_width, bbox_height = bbox

    return classify_frame_with_bbox(
        frame_bgr,
        bbox_left,
        bbox_top,
        bbox_width,
        bbox_height,
        sam_predictor,
        feature_extractor,
        scaler,
        svm_clf,
        device,
    )


def load_full_pipeline(device):
    detector = YOLO(os.path.join(MODEL_DIR, "hold_detector.pt"))

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    feature_extractor, scaler, svm_clf = load_model_artifacts(device)

    return detector, sam_predictor, feature_extractor, scaler, svm_clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default=os.path.join(RAW_IMAGES_DIR, "IMG_5078.jpg"))
    parser.add_argument("--yolo", action="store_true")
    args = parser.parse_args()
    img_path = args.img_path
    is_yolo = args.yolo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector, sam_predictor, feature_extractor, scaler, svm_clf = load_full_pipeline(device)

    frame_bgr = cv2.imread(img_path)
    bbox_left, bbox_top, bbox_width, bbox_height = get_bbox_for_input_img(img_path)

    if is_yolo:
        result = classify_frame_with_yolo(
            frame_bgr,
            detector,
            sam_predictor,
            feature_extractor,
            scaler,
            svm_clf,
            device,
        )
    else:
        result = classify_frame_with_bbox(
            frame_bgr,
            bbox_left,
            bbox_top,
            bbox_width,
            bbox_height,
            sam_predictor=sam_predictor,
            feature_extractor=feature_extractor,
            scaler=scaler,
            svm_clf=svm_clf,
            device=device,
        )

    print(CLASSIFICATIONS[result["prediction"]])
    print(result["bbox"])

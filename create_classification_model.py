import os
import cv2
import json
import numpy as np
import argparse
import joblib
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from segment_anything import sam_model_registry, SamPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from constants import (
    IMAGES_DIR,
    RAW_DIR,
    MODEL_DIR,
    PREPROCESSED_CACHE_IMAGES_DIR,
    PREPROCESSED_CACHE_MANIFEST,
    SEGMENTED_CACHE_IMAGES_DIR,
    SEGMENTED_CACHE_MANIFEST,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
    SPLIT_MANIFEST_PATH
)

class HoldPreprocessedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.to_tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["img_src"])

        if img is None:
            raise FileNotFoundError(f"Could not read img {sample["img_src"]}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = self.transform(img)
        label_tensor = torch.tensor(sample["label_value"], dtype=torch.long)

        return img_tensor, label_tensor


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)   # use BatchNorm to normalize inputs of each layer and speed up training
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4_drop = nn.Dropout2d(0.25)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(128)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2))
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.bn4(self.conv4(x))), kernel_size=2))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)
    
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.feature_extractor = CNNFeatureExtractor()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x
    
def load_img(img_src):
    img = cv2.imread(img_src)
    return img

def read_json_data(filename):
    with open(filename) as f:
        data = json.load(f)
        return data

# preprocessing and saving images
def build_segmented_cache(data, predictor):
    os.makedirs(SEGMENTED_CACHE_IMAGES_DIR, exist_ok=True)

    manifest = []

    for i, record in enumerate(data):
        img_src = os.path.join(RAW_DIR, IMAGES_DIR, record["filename"])

        img, metadata = preprocess_one_image(
            img_src=img_src,
            bbox_left=int(record["bbox_left"]),
            bbox_top=int(record["bbox_top"]),
            bbox_height=int(record["bbox_height"]),
            bbox_width=int(record["bbox_width"]),
            rotation=int(record["exif_rotation"]),
            label_name=record["label_name"],
            label_value=int(record["label_value"])
        )
        #print(f"preprocessed {img_src}")

        img, _ = segment_one_image(
            img,
            predictor,
            metadata["new_h"],
            metadata["new_w"],
            metadata["y_offset"],
            metadata["x_offset"],
        )
        #print(f"segmented {img_src}")

        base_name = os.path.basename(record["filename"][:-4])
        out_filename = f"{i:05d}_{base_name}.png"
        out_path = os.path.join(SEGMENTED_CACHE_IMAGES_DIR, out_filename)

        cv2.imwrite(out_path, img)

        manifest.append({
            "img_src": out_path,
            "label_value": int(record["label_value"]),
            "label_name": record["label_name"],
            "original_filename": record["filename"],
        })

    with open(SEGMENTED_CACHE_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_segmented_cache():
    with open(SEGMENTED_CACHE_MANIFEST) as f:
        return json.load(f)

def segmented_cache_exists():
    return os.path.exists(SEGMENTED_CACHE_MANIFEST)


def build_preprocessed_cache(data):
    os.makedirs(PREPROCESSED_CACHE_IMAGES_DIR, exist_ok=True)
    manifest = []
    for i, record in enumerate(data):
        img_src = os.path.join(RAW_DIR, IMAGES_DIR, record["filename"])

        img, _ = preprocess_one_image(
            img_src=img_src,
            bbox_left=int(record["bbox_left"]),
            bbox_top=int(record["bbox_top"]),
            bbox_height=int(record["bbox_height"]),
            bbox_width=int(record["bbox_width"]),
            rotation=int(record["exif_rotation"]),
            label_name=record["label_name"],
            label_value=int(record["label_value"])
        )

        # Save regular preprocessed images too, if desired:
        base_name = os.path.basename(record["filename"][:-4])
        out_filename = f"{i:05d}_{base_name}.png"
        out_path = os.path.join(PREPROCESSED_CACHE_IMAGES_DIR, out_filename)
        cv2.imwrite(out_path, img)

        manifest.append({
            "img_src": out_path,
            "label_value": int(record["label_value"]),
            "label_name": record["label_name"],
            "original_filename": record["filename"],
        })
    
    with open(PREPROCESSED_CACHE_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest

def load_preprocessed_cache():
    with open(PREPROCESSED_CACHE_MANIFEST) as f:
        return json.load(f)
    
def preprocessed_cache_exists():
    return os.path.exists(PREPROCESSED_CACHE_MANIFEST)


def split_samples_from_manifest(samples, split_manifest_path):
    with open(split_manifest_path, "r") as f:
        split_manifest = json.load(f)

    train_filenames = set(split_manifest["train"])
    val_filenames = set(split_manifest["val"])
    test_filenames = set(split_manifest["test"])

    train_samples = []
    val_samples = []
    test_samples = []

    skipped_samples = []

    for sample in samples:
        original_filename = sample["original_filename"]

        if original_filename in train_filenames:
            train_samples.append(sample)
        elif original_filename in val_filenames:
            val_samples.append(sample)
        elif original_filename in test_filenames:
            test_samples.append(sample)
        else:
            skipped_samples.append(sample)

    if skipped_samples:
        print(f"Warning: skipped {len(skipped_samples)} samples not found in split manifest.")

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    return train_samples, val_samples, test_samples


def clamp_bbox(left: int, top: int, width: int, height: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    left = max(0, min(left, img_w))
    top = max(0, min(top, img_h))
    right = max(left, min(left + width, img_w))
    bottom = max(top, min(top + height, img_h))
    return left, top, right - left, bottom - top

def pad_and_clamp_bbox(left: int, top: int, width: int, height: int, pad: int, img_w: int, img_h: int):
    new_left = max(0, left - pad)
    new_top = max(0, top - pad)
    new_right = min(img_w, left + width + pad)
    new_bottom = min(img_h, top + height + pad)
    return new_left, new_top, new_right - new_left, new_bottom - new_top

def crop_image(
    img_src: str,
    bbox_left: int,
    bbox_top: int,
    bbox_width: int,
    bbox_height: int, 
    rotation: int,
    label_name: str,
    label_value: int,
    pad: int = 4
):
    img = Image.open(img_src)

    if img.mode == "RGBA":
        img = img.convert("RGB")

    raw_w, raw_h = img.size
    left, top, width, height = clamp_bbox(bbox_left, bbox_top, bbox_width, bbox_height, raw_w, raw_h)
    left, top, width, height = pad_and_clamp_bbox(left, top, width, height, pad, raw_w, raw_h)
    out = {
        "bbox_left": left,
        "bbox_top": top, 
        "bbox_width": width,
        "bbox_height": height,
        "raw_w": raw_w,
        "raw_h": raw_h,
        "rotation": rotation,
        "label_name": label_name,
        "label_value": label_value
    }

    cropped = img.crop((left,top,left+width,top+height)) 

    if rotation == 3:
        cropped = cropped.rotate(180, expand=True)
    elif rotation == 6:
        cropped = cropped.rotate(-90, expand=True)
    elif rotation == 1:
        pass
    else:
        raise ValueError(f"Unsupported exif_rotation: {rotation}")
    
    cropped = np.array(cropped)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    return cropped, out

def normalize_img_size(img, size=256):
    h, w = img.shape[:2]

    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    resize_meta = {
        "canvas_size": size,
        "new_h": new_h,
        "new_w": new_w,
        "y_offset": y_offset,
        "x_offset": x_offset,
    }
    return canvas, resize_meta


def clahe_lighting_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_l = clahe.apply(l)
    lab_merged = cv2.merge((clahe_l, a, b))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

def preprocess_one_image(
    img_src: str,
    bbox_left: int,
    bbox_top: int,
    bbox_width: int,
    bbox_height: int,
    rotation: int,
    label_name: str,
    label_value: int,
    output_size: int = 256,
    pad: int = 4,
):
    cropped, metadata = crop_image(
        img_src,
        bbox_left,
        bbox_top,
        bbox_width,
        bbox_height, 
        rotation,
        label_name,
        label_value,
        pad
    )
    normalized, resize_meta = normalize_img_size(cropped, size=output_size)
    equalized = clahe_lighting_bgr(normalized)

    merged_metadata = metadata | resize_meta
    return equalized, merged_metadata

def get_roi_from_preprocessed(img, new_h, new_w, h_offset, w_offset):
    return img[h_offset:h_offset+new_h, w_offset:w_offset+new_w]

def get_image_segmentation(img, mask, h_offset, w_offset):
    result_img = img.copy()
    result_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask_h, mask_w = mask.shape

    roi = result_img[h_offset:h_offset+mask_h, w_offset:w_offset+mask_w]
    roi[mask == 0] = (0,0,0) 
    result_mask[h_offset:h_offset+mask_h, w_offset:w_offset+mask_w] = mask

    return result_img, result_mask

def get_segmentation_prompts(roi_w, roi_h, pad):
    prompts = {}
    
    center_coords = np.array([
        [roi_w // 2, roi_h // 2],
        [roi_w // 2 + pad, roi_h // 2],
        [roi_w // 2 - pad, roi_h // 2],
        [roi_w // 2, roi_h // 2 + pad],
        [roi_w // 2, roi_h // 2 - pad],
    ], dtype=np.float32)
    center_labels = np.array([1,1,1,1,1])
    prompts["center"] = (center_coords, center_labels)

    neg_corner_coords = np.array([
        [roi_w // 2, roi_h // 2],
        [roi_w // 2 + pad, roi_h // 2],
        [roi_w // 2 - pad, roi_h // 2],
        [pad, pad],
        [roi_w - 1 - pad, pad],
        [pad, roi_h - 1 - pad],
        [roi_w - 1 - pad, roi_h - 1 - pad]
    ], dtype=np.float32)
    neg_corner_labels = np.array([1,1,1,0,0,0,0])
    prompts["neg_corner"] = (neg_corner_coords, neg_corner_labels)

    # neg_border_midpt_coords = np.array([
    #     [roi_w // 2, roi_h // 2],
    #     [roi_w // 2 + pad, roi_h // 2],
    #     [roi_w // 2 - pad, roi_h // 2],
    #     [roi_w // 2, pad],
    #     [pad, roi_h // 2],
    #     [roi_w // 2, roi_h - 1 - pad],
    #     [roi_w - 1 - pad, roi_h // 2]
    # ], dtype=np.float32)
    # neg_border_midpt_labels = np.array([1,1,1,0,0,0,0])
    # prompts["neg_border_midpt"] = (neg_border_midpt_coords, neg_border_midpt_labels)

    # r = min(roi_w, roi_h) // 5
    # center_ring_coords = np.array([
    #     [roi_w // 2, roi_h // 2 - r],
    #     [roi_w // 2 + r, roi_h // 2],
    #     [roi_w // 2, roi_h // 2 + r],
    #     [roi_w // 2 - r, roi_h // 2],
    # ], dtype=np.float32)
    # center_ring_labels = np.array([1,1,1,1])
    # prompts["center_ring"] = (center_ring_coords, center_ring_labels)

    # prompts["bounding_box"] = (None, None)

    return prompts

def mask_has_center_region(mask, roi_w, roi_h, radius=25, min_ratio=0.1):
    center_x, center_y = roi_w // 2, roi_h // 2

    y_top = max(0, center_y - radius)
    y_bottom = min(roi_h, center_y + radius)
    x_left = max(0, center_x - radius)
    x_right = min(roi_w, center_x + radius)

    center_patch = mask[y_top:y_bottom, x_left:x_right]
    return np.mean(center_patch) >= min_ratio

def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask
    
    largest_label = 1 + np.argmax(stats[1: , cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255

def segment_one_image(img, predictor, new_h, new_w, h_offset, w_offset, pad=5):
    roi = get_roi_from_preprocessed(img, new_h, new_w, h_offset, w_offset)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    predictor.set_image(roi_rgb)

    roi_h, roi_w = roi.shape[:2]
    input_box = np.array([0,0, roi_w - 1, roi_h - 1], dtype=np.float32)

    prompts = get_segmentation_prompts(roi_w, roi_h, pad)

    best_score = -1
    best_mask = None
    best_prompt = None
    fallback_ranges = [
        (0.20, 0.80, 0.10),
        (0.10, 0.90, 0.05),
        (0.05, 0.95, 0.00),
    ]

    for lower, upper, center_min_ratio in fallback_ranges:
        for prompt_name, (prompt_coords, prompt_labels) in prompts.items():
            masks, scores, _ = predictor.predict(
                point_coords=prompt_coords,
                point_labels=prompt_labels,
                box=input_box[None, :],
                multimask_output=True
            )

            for mask, score in zip(masks, scores):
                area_ratio = np.mean(mask)

                if area_ratio < lower or area_ratio > upper:
                    continue

                if not mask_has_center_region(mask, roi_w, roi_h, min_ratio=center_min_ratio):
                    continue

                if score > best_score:
                    best_score = score
                    best_mask = mask
                    best_prompt = prompt_name
                    
        if best_mask is not None:
            print(f"Using {best_prompt}, score={best_score}, range=({lower}, {upper})")
            break

    if best_mask is None:
        # choose highest SAM score with no filtering
        best_score = -1
        for prompt_name, (prompt_coords, prompt_labels) in prompts.items():
            masks, scores, _ = predictor.predict(
                point_coords=prompt_coords,
                point_labels=prompt_labels,
                box=input_box[None, :],
                multimask_output=True
            )

            i = int(np.argmax(scores))
            if scores[i] > best_score:
                best_score = scores[i]
                best_mask = masks[i]
                best_prompt = prompt_name

        print(f"LAST RESORT: Using {best_prompt}, score={best_score}")

    mask = (best_mask > 0).astype(np.uint8) * 255
    mask = keep_largest_component(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    # mask = fill_holes(mask)

    segmented_img, resized_mask = get_image_segmentation(img, mask, h_offset, w_offset)
    return segmented_img, resized_mask

# def segment_one_image(img, predictor, new_h, new_w, h_offset, w_offset, pad=5):
#     roi = get_roi_from_preprocessed(img, new_h, new_w, h_offset, w_offset)
#     roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#     predictor.set_image(roi_rgb)

#     roi_h, roi_w = roi.shape[:2]
#     input_box = np.array([0,0, roi_w - 1, roi_h - 1], dtype=np.float32)

#     point_coords = np.array([
#         [roi_w // 2, roi_h // 2],
#         [roi_w // 2 + pad, roi_h // 2],
#         [roi_w // 2 - pad, roi_h // 2],
#         [pad, pad],
#         [roi_w - 1 - pad, pad],
#         [pad, roi_h - 1 - pad],
#         [roi_w - 1 - pad, roi_h - 1 - pad]
#     ], dtype=np.float32)
#     point_labels = np.array([1,1,1,0,0,0,0])

#     masks, scores, _ = predictor.predict(
#         point_coords=point_coords,
#         point_labels=point_labels,
#         box = input_box[None, :],
#         multimask_output=True
#     )

#     max_score_index = int(np.argmax(scores))
#     max_score_mask = masks[max_score_index]

#     mask = (max_score_mask > 0).astype(np.uint8) * 255
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
#     # mask = fill_holes(mask)

#     segmented_img, resized_mask = get_image_segmentation(img, mask, h_offset, w_offset)
#     return segmented_img, resized_mask

def get_cnn_features_SVM(model, dataloader, device):
    model.eval()
    model.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # print(f"Feature batch {batch_idx + 1}/{len(dataloader)}")
            images = images.to(device)
            features = model(images)
            features = features.flatten(start_dim=1)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_features), np.concatenate(all_labels)


def build_feature_extractor_RESNET():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])  # remove final FC
    return model


def train_model(model, train_loader, val_loader, device, epochs=10, train_layer_4=False, train_all_layers=False):
    model.to(device);
    optimizer = torch.optim.Adam(
        list(model.fc.parameters()),
        lr=1e-3
    )
    for param in model.parameters():
        param.requires_grad = False


    if train_layer_4:
        for param in model.layer4.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            list(model.layer4.parameters()) + list(model.fc.parameters()),
            lr=1e-4
        )
    
    if train_all_layers:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-5
        )

    for param in model.fc.parameters():
        param.requires_grad = True

    train_losses = []
    train_accs = []
    val_accs = []

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")    
    
    criterion = nn.CrossEntropyLoss()
    

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # print(f"Epoch {epoch + 1}/{epochs}, batch {batch_idx + 1}/{len(train_loader)}")
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        correct = 0
        total = 0

        num_classes = 7
        class_correct = [0 for _ in range(num_classes)]
        class_total = [0 for _ in range(num_classes)]

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model.forward(images)                     # returns (batch_size, 10) holding the output prob for every class in every image
                predictions = torch.argmax(outputs, dim=1)          # grabs the max prob reducing along each row to get (batch_size, 1)

                # per-class accuracy
                for i in range(len(labels)):
                    label = labels[i]
                    pred = predictions[i]
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

                correct += torch.sum(predictions == labels).float() # sums the correct predictions and turns to float for calculation later
                total += images.shape[0]                             # grabs the total number of images in this batch

        val_acc = 100. * correct.item() / total

        per_class_acc = []
        for c in range(num_classes):
            if class_total[c] > 0:
                acc = 100.0 * class_correct[c] / class_total[c]
            else:
                acc = 0.0
            per_class_acc.append(acc)
        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, '
            f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        print(f"Per-class accuracy:", end=" ")
        for c, acc in enumerate(per_class_acc):
            print(f"  Class {c}: {acc:.2f}%", end="  |  ")
        print("\n")
    
    print(f"Average Val Accuracy: {np.mean(val_accs):.2f}")
    
    return train_losses, train_accs, val_accs


def run_svm_classification(X_train, Y_train, X_val, Y_val, kernel="rbf"):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Change to linear to see other results
    clf = svm.SVC(kernel=kernel, C=1.0, gamma='scale')
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_val)

    accuracy = np.mean(Y_pred == Y_val)
    print(f"SVM Classification Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(Y_val, Y_pred)}")

    # UNCOMMENT ONLY WHEN DOING FINAL TEST ON CHOSEN MODEL
    # Y_test_pred = clf.predict(X_test)
    # test_accuracy = np.mean(Y_test_pred == Y_test)
    # print(f"Test Accuracy: {test_accuracy:.4f}")

    return clf, scaler, Y_pred



if __name__ == "__main__":
    num_classes = 7
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--num_epochs", default=20)
    args = parser.parse_args()
    num_epochs = int(args.num_epochs) if args.num_epochs else 20
    print(f"starting with model {args.model}")

    data_path = os.path.join(RAW_DIR, "annotations.json")
    data = read_json_data(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessed_data = []

    is_sam_model = args.model in ["3", "4", "7"]  
    predictor = None

    if is_sam_model:
        if segmented_cache_exists():
            print("Loading segmented cache")
            preprocessed_data = load_segmented_cache()
        else:
            print("Building segmented cache")
            sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            manifest = build_segmented_cache(data, predictor)
            preprocessed_data = load_segmented_cache();
    else:
        if preprocessed_cache_exists():
            print("Loading preprocessed cache")
            preprocessed_data = load_preprocessed_cache()
        else:
            print("Building preprocessed cache")
            manifest = build_preprocessed_cache(data)
            preprocessed_data = load_preprocessed_cache()

    train_samples, val_samples, test_samples = split_samples_from_manifest(
        preprocessed_data,
        SPLIT_MANIFEST_PATH
    )

    train_dataset = HoldPreprocessedDataset(samples=train_samples)
    val_dataset = HoldPreprocessedDataset(samples=val_samples)
    test_dataset = HoldPreprocessedDataset(samples=test_samples)

    batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    #### MAIN MODEL: HAS HIGHEST ACCURACY AFTER MANY TRAINING PASSES
    if args.model == "4":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        train_losses, train_accs, val_accs = train_model(
            model, 
            train_loader, 
            val_loader, 
            device, 
            epochs=num_epochs, 
            train_layer_4=True
        )
    
        # Get the feature extractor trained from ResNet18 by cutting off its FC layer
        feature_extractor = nn.Sequential(*list(model.children())[:-1]) 

        # Get features for every image in training data
        X_train, Y_train = get_cnn_features_SVM(
            feature_extractor, 
            train_loader, 
            device=device
        )
        
        #Get features for every image in validation data
        X_val, Y_val = get_cnn_features_SVM(
            feature_extractor, 
            val_loader, 
            device=device
        )

        # Classify feature vectors for every img via SVM & determine accuracy
        clf, scaler, Y_pred = run_svm_classification(
            X_train, 
            Y_train, 
            X_val, 
            Y_val, 
            kernel="rbf"
        )

        os.makedirs(MODEL_DIR, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model4_resnet.pt"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "model4_scaler.joblib"))
        joblib.dump(clf, os.path.join(MODEL_DIR, "model4_svm.joblib"))

        print("Saved model 4 artifacts:")
        print("  models/model4_resnet.pt")
        print("  models/model4_scaler.joblib")
        print("  models/model4_svm.joblib")


    #### Other Models used when attempting to determine best model for this project

    # SVM linear kernel classification
    if args.model == "3":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        train_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, device, epochs=num_epochs, train_layer_4=True)
    
        feature_extractor = nn.Sequential(*list(model.children())[:-1]) 
        X_train, Y_train = get_cnn_features_SVM(feature_extractor, train_loader, device=device)
        X_val, Y_val = get_cnn_features_SVM(feature_extractor, val_loader, device=device)

        clf, Y_pred = run_svm_classification(X_train, Y_train, X_val, Y_val, kernel="linear")

    # Direct classification via ResNet18
    if args.model == "6":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        train_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, device, epochs=10)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fcTrained.pt"))
        train_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, device, epochs=10, train_layer_4=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "layer4Trained.pt"))
        train_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, device, epochs=10, train_all_layers=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "allTrained.pt"))

    # Custom CNN classifier using SAM for segmentation
    if args.model == "7":
        model = CNNClassifier(num_classes).to(device)
        train_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, device, epochs=50, train_all_layers=True)
        feature_extractor = model.feature_extractor

        X_train, y_train = get_cnn_features_SVM(feature_extractor, train_loader, device)
        X_val, y_val = get_cnn_features_SVM(feature_extractor, val_loader, device)

        clf, y_pred = run_svm_classification(X_train, y_train, X_val, y_val, kernel="linear")

    if args.model == "8":
        model = CNNClassifier(num_classes).to(device)
        train_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, device, epochs=50, train_all_layers=True)

    
    



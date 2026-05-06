import os
import json
from pathlib import Path
from urllib.parse import urlparse
import requests

INPUT_NDJSON = "annotations.ndjson"
IMAGES_DIR = "prev_images"
OUTPUT_JSON = "prev_annotations.json"

os.makedirs(IMAGES_DIR, exist_ok=True)


def safe_filename(name: str) -> str:
    return os.path.basename(name).strip()


def get_extension_from_url(url: str, default=".jpg") -> str:
    path = urlparse(url).path
    ext = Path(path).suffix
    return ext if ext else default


def iter_ndjson(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_label_info(obj):
    for cls in obj.get("classifications", []):
        radio = cls.get("radio_answer")
        if radio:
            return {
                "label_name": radio.get("name"),
                "label_value": radio.get("value"),
            }
    return {
        "label_name": None,
        "label_value": None,
    }


def download_image(image_url: str, output_path: str):
    if os.path.exists(output_path):
        return

    response = requests.get(image_url, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)


def main():
    bbox_records = []
    downloaded = set()

    for record in iter_ndjson(INPUT_NDJSON):
        data_row = record.get("data_row", {})
        media = record.get("media_attributes", {})

        external_id = data_row.get("external_id")
        image_url = data_row.get("row_data")

        if not external_id or not image_url:
            continue

        filename = safe_filename(external_id)
        if not Path(filename).suffix:
            filename += get_extension_from_url(image_url)

        local_image_path = os.path.join(IMAGES_DIR, filename)

        if filename not in downloaded:
            try:
                download_image(image_url, local_image_path)
                downloaded.add(filename)
                print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue

        img_w = media.get("width")
        img_h = media.get("height")
        exif_rotation = media.get("exif_rotation", 1)

        for project_data in record.get("projects", {}).values():
            for label_entry in project_data.get("labels", []):
                objects = label_entry.get("annotations", {}).get("objects", [])

                for obj in objects:
                    bbox = obj.get("bounding_box")
                    if not bbox:
                        continue

                    label_info = extract_label_info(obj)

                    bbox_records.append({
                        "filename": filename,
                        "width": int(img_w) if img_w is not None else None,
                        "height": int(img_h) if img_h is not None else None,
                        "label_name": label_info["label_name"],
                        "label_value": label_info["label_value"],
                        "bbox_left": int(round(bbox.get("left", 0))),
                        "bbox_top": int(round(bbox.get("top", 0))),
                        "bbox_width": int(round(bbox.get("width", 0))),
                        "bbox_height": int(round(bbox.get("height", 0))),
                        "exif_rotation": int(exif_rotation) if exif_rotation is not None else 1,
                    })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(bbox_records, f, indent=2)

    print(f"\nDone.")
    print(f"Downloaded {len(downloaded)} images into '{IMAGES_DIR}/'")
    print(f"Wrote {len(bbox_records)} bounding-box records to '{OUTPUT_JSON}'")


if __name__ == "__main__":
    main()
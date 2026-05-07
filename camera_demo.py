# camera_demo.py

import argparse
import cv2
import torch

from main import (
    load_full_pipeline,
    classify_frame_with_yolo,
)

from constants import CLASSIFICATIONS


LIVE_WINDOW = "Live Camera"
CLASSIFICATION_WINDOW = "Last Classification"


def draw_result(frame_bgr, result):
    if result is None:
        cv2.putText(
            frame_bgr,
            "No hold detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return frame_bgr

    bbox = result["bbox"]
    prediction = CLASSIFICATIONS[result["prediction"]]

    left = int(bbox["left"])
    top = int(bbox["top"])
    width = int(bbox["width"])
    height = int(bbox["height"])

    right = left + width
    bottom = top + height

    cv2.rectangle(
        frame_bgr,
        (left, top),
        (right, bottom),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame_bgr,
        f"{prediction}",
        (left, max(30, top - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    return frame_bgr


def window_is_open(window_name):
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    last_classified_frame = None
    classification_window_open = False

    print("Camera started.")
    print("Press SPACE to classify current frame.")
    print("Press C to close the classification window.")
    print("Press Q to quit.")

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        live_frame = frame_bgr.copy()
        cv2.imshow(LIVE_WINDOW, live_frame)

        # Only show the still frame if the classification window is meant to be open.
        if classification_window_open and last_classified_frame is not None:
            if window_is_open(CLASSIFICATION_WINDOW):
                cv2.imshow(CLASSIFICATION_WINDOW, last_classified_frame)
            else:
                classification_window_open = False

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            if window_is_open(CLASSIFICATION_WINDOW):
                cv2.destroyWindow(CLASSIFICATION_WINDOW)
            classification_window_open = False
            last_classified_frame = None

        if key == 32:  # SPACE
            print("Loading model pipeline...")

            try:
                (
                    detector,
                    sam_predictor,
                    feature_extractor,
                    scaler,
                    svm_clf,
                ) = load_full_pipeline(device)

                print("Classifying frame...")

                captured_frame = frame_bgr.copy()

                result = classify_frame_with_yolo(
                    captured_frame,
                    detector,
                    sam_predictor,
                    feature_extractor,
                    scaler,
                    svm_clf,
                    device,
                )

                last_classified_frame = draw_result(captured_frame.copy(), result)
                classification_window_open = True

                cv2.imshow(CLASSIFICATION_WINDOW, last_classified_frame)

                if result is None:
                    print("No hold detected.")
                else:
                    print(f"Prediction: {CLASSIFICATIONS[result['prediction']]}")
                    print(f"BBox: {result['bbox']}")

            except Exception as e:
                print(f"Classification failed: {e}")

                failed_frame = frame_bgr.copy()
                cv2.putText(
                    failed_frame,
                    "Classification failed",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

                last_classified_frame = failed_frame
                classification_window_open = True
                cv2.imshow(CLASSIFICATION_WINDOW, last_classified_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
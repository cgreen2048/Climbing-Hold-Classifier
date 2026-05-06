# camera_demo.py

import cv2

from main import (
    load_full_pipeline,
    classify_frame_with_yolo,
)


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
    prediction = result["prediction"]

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
        f"Class {prediction}",
        (left, max(30, top - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    return frame_bgr


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    last_result = None

    print("Camera started.")
    print("Press SPACE to classify current frame.")
    print("Press Q to quit.")

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        display_frame = frame_bgr.copy()

        if last_result is not None:
            display_frame = draw_result(display_frame, last_result)

        cv2.imshow("Climbing Hold Classifier", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:  # SPACE
            print("Loading model pipeline...")

            try:
                (
                    detector,
                    sam_predictor,
                    feature_extractor,
                    scaler,
                    svm_clf,
                    device,
                ) = load_full_pipeline()

                print("Classifying frame...")

                last_result = classify_frame_with_yolo(
                    frame_bgr,
                    detector,
                    sam_predictor,
                    feature_extractor,
                    scaler,
                    svm_clf,
                    device,
                )

                if last_result is None:
                    print("No hold detected.")
                else:
                    print(f"Prediction: {last_result['prediction']}")
                    print(f"BBox: {last_result['bbox']}")

            except Exception as e:
                print(f"Classification failed: {e}")
                last_result = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
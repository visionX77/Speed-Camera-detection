
from ultralytics import YOLO
import cv2
import easyocr
import time
import csv
import os

# ================= CONFIGURATION =================

VIDEO_PATH = "videos/road.mp4"
SPEED_LIMIT_KMH = 60

# IMPORTANT:
# Set this to the REAL distance in meters between
# the two measurement lines in your camera view.
DISTANCE_METERS = 10.0

LINE1_RATIO = 0.40
LINE2_RATIO = 0.60

OUTPUT_DIR = "result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================

print("Loading vehicle detector...")
vehicle_model = YOLO("yolo11n.pt")

print("Loading license plate OCR...")
# English OCR can read many Latin letters and digits.
# Korean plate recognition may require a suitable model
# or additional preprocessing for reliable results.
ocr_reader = easyocr.Reader(["en"], gpu=False)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO_PATH}")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    print("ERROR: Invalid video FPS.")
    cap.release()
    raise SystemExit

# Vehicle classes in the COCO model:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

previous_y = {}
crossing_times = {}
vehicle_speeds = {}
plate_texts = {}
last_ocr_time = {}

# Prevent repeated event records for the same track.
recorded_violations = set()

csv_path = os.path.join(OUTPUT_DIR, "violations.csv")

if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Video time (seconds)",
            "Vehicle ID",
            "License plate",
            "Estimated speed (km/h)",
            "Speed limit (km/h)"
        ])

frame_number = 0

print("VisionX speed detection started.")
print(f"Speed limit: {SPEED_LIMIT_KMH} km/h")
print(f"Calibration distance: {DISTANCE_METERS} m")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    video_time = frame_number / fps

    height, width = frame.shape[:2]
    line1_y = int(height * LINE1_RATIO)
    line2_y = int(height * LINE2_RATIO)

    result = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        verbose=False
    )[0]

    output = frame.copy()

    cv2.line(output, (0, line1_y), (width, line1_y),
             (255, 255, 0), 2)
    cv2.line(output, (0, line2_y), (width, line2_y),
             (255, 255, 0), 2)

    cv2.putText(output, "LINE 1", (10, line1_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(output, "LINE 2", (10, line2_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    boxes = result.boxes

    if boxes is not None and boxes.id is not None:
        coords = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(
            coords, track_ids, class_ids
        ):
            x1, y1, x2, y2 = map(int, box)

            # Bottom-center of vehicle bounding box
            center_y = y2

            if track_id not in crossing_times:
                crossing_times[track_id] = {}

            old_y = previous_y.get(track_id)

            if old_y is not None:
                crossed_line1 = (
                    (old_y < line1_y <= center_y)
                    or (old_y > line1_y >= center_y)
                )
                crossed_line2 = (
                    (old_y < line2_y <= center_y)
                    or (old_y > line2_y >= center_y)
                )

                if crossed_line1:
                    crossing_times[track_id][1] = video_time

                if crossed_line2:
                    crossing_times[track_id][2] = video_time

                times = crossing_times[track_id]

                if 1 in times and 2 in times:
                    elapsed = abs(times[2] - times[1])

                    if elapsed > 0:
                        speed = DISTANCE_METERS / elapsed * 3.6
                        vehicle_speeds[track_id] = speed

                    crossing_times[track_id] = {}

            previous_y[track_id] = center_y

            speed = vehicle_speeds.get(track_id)
            speeding = (
                speed is not None and speed > SPEED_LIMIT_KMH
            )

            color = (0, 0, 255) if speeding else (0, 255, 0)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # OCR attempt on a vehicle crop.
            # This is only a prototype: it may read text on the car
            # rather than the actual license plate.
            if (
                track_id not in plate_texts
                and video_time - last_ocr_time.get(track_id, -10) >= 2
            ):
                last_ocr_time[track_id] = video_time

                vehicle_crop = frame[
                    max(0, y1):min(height, y2),
                    max(0, x1):min(width, x2)
                ]

                if vehicle_crop.size > 0:
                    try:
                        ocr_results = ocr_reader.readtext(
                            vehicle_crop,
                            detail=0
                        )

                        candidates = [
                            text.strip()
                            for text in ocr_results
                            if len(text.strip()) >= 4
                        ]

                        if candidates:
                            plate_texts[track_id] = candidates[0]
                    except Exception as exc:
                        print("OCR error:", exc)

            plate = plate_texts.get(track_id, "Not detected")

            if speed is None:
                speed_label = "Measuring..."
            else:
                speed_label = f"{speed:.1f} km/h"

            label = f"ID {track_id} | {speed_label}"
            cv2.putText(
                output, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

            cv2.putText(
                output, f"Plate: {plate}", (x1, min(height - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

            if speeding:
                cv2.putText(
                    output, "WARNING: SPEEDING",
                    (x1, min(height - 35, y2 + 45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
                )

                if track_id not in recorded_violations:
                    recorded_violations.add(track_id)

                    with open(
                        csv_path, "a", newline="", encoding="utf-8-sig"
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            round(video_time, 2),
                            track_id,
                            plate,
                            round(speed, 1),
                            SPEED_LIMIT_KMH
                        ])

                    image_path = os.path.join(
                        OUTPUT_DIR,
                        f"violation_{track_id}_{frame_number}.jpg"
                    )
                    cv2.imwrite(image_path, output)

    cv2.rectangle(output, (10, 10), (360, 45), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"SPEED LIMIT: {SPEED_LIMIT_KMH} km/h",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    cv2.imshow("VisionX - Speed and Plate Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("VisionX stopped.")
print(f"Violation records saved to: {csv_path}")

from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open road video
video_path = "videos/road.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("VisionX vehicle detection started.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    # Detect and track vehicles
    results = model.track(
        frame,
        persist=True,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    # Draw boxes and IDs
    output = results[0].plot()

    # Display
    cv2.imshow("VisionX - Vehicle Detection", output)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("VisionX detection stopped.")
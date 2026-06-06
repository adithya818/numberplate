
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Optional: Set camera resolution
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO inference
    results = model(frame)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

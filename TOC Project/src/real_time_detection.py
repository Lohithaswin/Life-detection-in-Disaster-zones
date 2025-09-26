from ultralytics import YOLO
import cv2

def process_video(video_source, model_path="src/yolov8n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return []

    detected_classes = []
    frames_after_detection = 30  # Reduced extension: ~2 seconds (assuming 30 FPS)
    frame_delay = 33  # Frame delay in milliseconds for ~30 FPS

    detected_person = False  # Flag to track if "person" is detected
    persons_in_frame = []  # List to store person detections

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run the YOLO model on the current frame
            results = model.predict(source=frame, show=False)
            detections = results[0].boxes

            if detections:
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    detected_classes.append(class_name)

                    # Use green box for "person" and red for others
                    color = (0, 255, 0) if class_name == "person" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if class_name == "person":
                        detected_person = True  # Set flag if "person" is detected
                        persons_in_frame.append((x1, y1, x2, y2))  # Store "person" box coordinates

            # Show the video feed with bounding boxes
            cv2.imshow("Video Feed", frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

            # If a "person" is detected, extend the video playback
            if detected_person:
                for _ in range(frames_after_detection):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Redraw bounding boxes and labels during the extension
                    results = model.predict(source=frame, show=False)
                    detections = results[0].boxes
                    if detections:
                        for box in detections:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            color = (0, 255, 0) if class_name == "person" else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Display the extended video feed
                    cv2.imshow("Video Feed", frame)
                    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                        break
                return detected_classes

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return detected_classes
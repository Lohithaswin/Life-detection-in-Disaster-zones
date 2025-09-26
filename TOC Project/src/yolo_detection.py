from ultralytics import YOLO

from ultralytics import YOLO

def detect_objects(image_path, model_path="src/yolov8n.pt"):
    model = YOLO(model_path)
    results = model.predict(source=image_path, save=True)
    detections = results[0].boxes

    detected_classes = []
    if detections:
        for box in detections:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.append(class_name)

    return detected_classes
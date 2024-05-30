# preprocess_and_detect.py
import fiftyone as fo
from ultralytics import YOLO

# Load the saved dataset
dataset = fo.load_dataset("coco-2017-train-1000")

# Initialize the YOLOv5 model
model = YOLO("yolov5su.pt")  # Ensure you have the correct model path

# Define the preprocessing and detection logic
for sample in dataset:
    # Perform any preprocessing needed
    image = sample.filepath
    results = model(image)
    
    # Extract detection data
    detections = []
    if isinstance(results, list):  # Handle list output
        results = results[0]
    for box in results.boxes.xyxy.cpu().numpy():  # Access the bounding boxes
        if len(box) >= 6:  # Ensure the box has sufficient length
            label = model.names[int(box[5])] if int(box[5]) < len(model.names) else "Unknown"  # Ensure correct label assignment
            bbox = box[0:4] / [sample.metadata.width, sample.metadata.height, sample.metadata.width, sample.metadata.height]
            confidence = box[4]
            
            detection = fo.Detection(
                label=label,
                bounding_box=bbox.tolist(),
                confidence=confidence.item()
            )
            detections.append(detection)
    
    sample["detections"] = fo.Detections(detections=detections)
    sample.save()

# Launch FiftyOne App to visualize the results
session = fo.launch_app(dataset, port=5151)
session.wait()

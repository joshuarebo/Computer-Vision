# run_detection.py
import fiftyone as fo
import fiftyone.zoo as foz
import torch
from pathlib import Path

# Set a specific port for FiftyOne's database service
import fiftyone.core.config as foc
foc.default_db_port = 5151

# Load a model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Download a sample dataset
dataset = foz.load_zoo_dataset("coco-2017", split="validation")

# Choose a sample image
image_path = dataset.first().filepath

# Perform inference
results = model(image_path)

# Display results
results.show()

# Debug prints
print(f"Results: {results}")

# Parse results
predictions = results.pred[0]  # Get the predictions from the first result
for pred in predictions:
    print(f"Prediction: {pred}")
    # Debug print for pred
    print(f"pred: {pred}")
    
    # Access cls, xyxy, and conf directly from tensor
    class_idx = int(pred[5].item())
    bbox = pred[:4].tolist()  # [x1, y1, x2, y2]
    confidence = pred[4].item()

    # Debug print for class index and result names
    print(f"class_idx: {class_idx}")
    print(f"result.names: {results.names}")

    # Fetch the label using the class index
    label = results.names[class_idx]
    print(f"Detected label: {label}")

    # Further processing if needed
    # ...

# View results in FiftyOne
session = fo.launch_app(dataset)
session.wait()

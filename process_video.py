import cv2
import torch
import numpy as np
import torchvision
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def process_video(input_video_path, output_video_path, log_file):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    with open(log_file, 'w') as log:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = model(frame)

            # Annotate frame with results
            for *xyxy, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)]
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Log prediction
                log.write(f"{frame_count} {label} {conf:.2f} {x1} {y1} {x2} {y2}\n")

            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    input_video_path = "sample_video.mp4"
    output_video_path = "output_yolo_video.mp4"
    log_file = "yolo_log.txt"
    process_video(input_video_path, output_video_path, log_file)

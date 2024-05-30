import cv2
import torch
import torchvision
from pathlib import Path

# Load Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='COCO_V1')
model.eval()

def process_video(input_video_path, output_video_path, log_file):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    with open(log_file, 'w') as log:
        while cap.isOpened() and frame_count < 300:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to reduce processing time
            small_frame = cv2.resize(frame, (320, 240))
            
            # Preprocess frame
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            img = transform(small_frame).unsqueeze(0)

            # Inference
            with torch.no_grad():
                results = model(img)

            # Annotate frame with results
            for box, label, score in zip(results[0]['boxes'], results[0]['labels'], results[0]['scores']):
                if score > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box.int().tolist()
                    x1 = int(x1 * (width / 320))
                    y1 = int(y1 * (height / 240))
                    x2 = int(x2 * (width / 320))
                    y2 = int(y2 * (height / 240))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Log prediction
                    log.write(f"{frame_count} {label} {score:.2f} {x1} {y1} {x2} {y2}\n")

            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    input_video_path = "sample_video.mp4"
    output_video_path = "output_maskrcnn_video.mp4"
    log_file = "maskrcnn_log.txt"
    process_video(input_video_path, output_video_path, log_file)

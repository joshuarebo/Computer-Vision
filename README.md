# Computer Vision Object Detection Project

## Overview
This project involves developing a computer vision system capable of real-time object detection and segmentation in video sequences using three state-of-the-art models: YOLOv5, Mask R-CNN, and Faster R-CNN. The system evaluates each model's performance with respect to validity, reliability, and objectivity on a representative dataset.

## Models Used
- **YOLOv5:** Implemented using the PyTorch framework. Known for its high speed and real-time processing capabilities.
- **Mask R-CNN:** Implemented using the Matterport Mask R-CNN library. Provides high accuracy and instance segmentation.
- **Faster R-CNN:** Implemented using TensorFlow’s Object Detection API. Balances between speed and accuracy.

## Dataset
The models are evaluated on the COCO dataset, which provides annotated images for object detection, segmentation, and keypoint detection tasks.

## Evaluation Metrics
The models are evaluated using the following metrics:
- Precision
- Recall
- F1-score
- Accuracy

## Results
- **Faster R-CNN:**
  - Average Score: 0.86
  - Precision: 1.00
  - Recall: 1.00
  - F1-score: 1.00
  - Accuracy: 1.00
- **Mask R-CNN:**
  - Average Score: 0.80
  - Precision: 1.00
  - Recall: 1.00
  - F1-score: 1.00
  - Accuracy: 1.00
- **YOLOv5:**
  - Average Score: 0.55
  - Precision: 0.61
  - Recall: 0.61
  - F1-score: 0.61
  - Accuracy: 0.44

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/joshuarebo/Computer-Vision.git
   cd Computer-Vision
Set up virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required dependencies:
pip install -r requirements.txt

Run the evaluation Script:
python run_evaluation.py
Video Demonstrations:
YOLOv5 Detection: [https://youtu.be/qD2jck24VgI]
- Mask R-CNN Detection: [https://youtu.be/c80Yog-Q3zg]
- Faster R-CNN Detection: [https://youtu.be/spnFwGoKi6Y]
Acknowledgments
COCO Dataset: Provided by Microsoft COCO, a large-scale object detection, segmentation, and captioning dataset.
YOLOv5: Developed by Joseph Redmon and Ali Farhadi.
Mask R-CNN: Developed by Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross B. Girshick.
Faster R-CNN: Developed by Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun.
OpenCV: An open-source computer vision and machine learning software library
import argparse
import json

def load_json(file_path):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)

def evaluate_model(data):
    total_frames = len(data)
    total_score = sum(item['score'] for item in data)
    average_score = total_score / total_frames
    return average_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance based on detection logs")
    parser.add_argument('--fasterrcnn', type=str, required=True, help="Path to Faster R-CNN JSON data")
    parser.add_argument('--maskrcnn', type=str, required=True, help="Path to Mask R-CNN JSON data")
    parser.add_argument('--yolo', type=str, required=True, help="Path to YOLO JSON data")
    args = parser.parse_args()

    fasterrcnn_data = load_json(args.fasterrcnn)
    maskrcnn_data = load_json(args.maskrcnn)
    yolo_data = load_json(args.yolo)

    fasterrcnn_score = evaluate_model(fasterrcnn_data)
    maskrcnn_score = evaluate_model(maskrcnn_data)
    yolo_score = evaluate_model(yolo_data)

    print(f"Faster R-CNN Average Score: {fasterrcnn_score:.2f}")
    print(f"Mask R-CNN Average Score: {maskrcnn_score:.2f}")
    print(f"YOLO Average Score: {yolo_score:.2f}")

if __name__ == "__main__":
    main()

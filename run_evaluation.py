import os
import json

def parse_fasterrcnn_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        frame_id = int(parts[0])
        class_id = int(parts[1])
        score = float(parts[2])
        bbox = list(map(int, parts[3:]))
        data.append({'frame_id': frame_id, 'class_id': class_id, 'score': score, 'bbox': bbox})
    return data

def parse_maskrcnn_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        frame_id = int(parts[0])
        class_id = int(parts[1])
        score = float(parts[2])
        bbox = list(map(int, parts[3:]))
        data.append({'frame_id': frame_id, 'class_id': class_id, 'score': score, 'bbox': bbox})
    return data

def parse_yolo_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        frame_id = int(parts[0])
        class_names = []
        index = 1
        while not parts[index].replace('.', '', 1).isdigit():
            class_names.append(parts[index])
            index += 1
        score = float(parts[index])
        bbox = list(map(int, parts[index + 1:]))
        data.append({'frame_id': frame_id, 'class_names': class_names, 'score': score, 'bbox': bbox})
    return data

def save_to_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def calculate_metrics(data):
    tp = sum(1 for item in data if item['score'] >= 0.5)
    fp = sum(1 for item in data if item['score'] < 0.5)
    fn = sum(1 for item in data if item['score'] < 0.5)
    tn = 0  # Assuming no true negatives in this context

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = tp / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

    return precision, recall, f1_score, accuracy

def main():
    fasterrcnn_log_path = 'fasterrcnn_log.txt'
    maskrcnn_log_path = 'maskrcnn_log.txt'
    yolo_log_path = 'yolo_log.txt'

    fasterrcnn_data = parse_fasterrcnn_log(fasterrcnn_log_path)
    maskrcnn_data = parse_maskrcnn_log(maskrcnn_log_path)
    yolo_data = parse_yolo_log(yolo_log_path)

    save_to_json(fasterrcnn_data, 'fasterrcnn_data.json')
    save_to_json(maskrcnn_data, 'maskrcnn_data.json')
    save_to_json(yolo_data, 'yolo_data.json')

    # Call evaluation script and print metrics
    os.system('python evaluation.py --fasterrcnn fasterrcnn_data.json --maskrcnn maskrcnn_data.json --yolo yolo_data.json')

    # Calculate metrics and print them
    fasterrcnn_precision, fasterrcnn_recall, fasterrcnn_f1, fasterrcnn_accuracy = calculate_metrics(fasterrcnn_data)
    maskrcnn_precision, maskrcnn_recall, maskrcnn_f1, maskrcnn_accuracy = calculate_metrics(maskrcnn_data)
    yolo_precision, yolo_recall, yolo_f1, yolo_accuracy = calculate_metrics(yolo_data)

    print(f"Faster R-CNN Metrics:")
    print(f"  Precision: {fasterrcnn_precision:.2f}")
    print(f"  Recall: {fasterrcnn_recall:.2f}")
    print(f"  F1-score: {fasterrcnn_f1:.2f}")
    print(f"  Accuracy: {fasterrcnn_accuracy:.2f}")
    
    print(f"Mask R-CNN Metrics:")
    print(f"  Precision: {maskrcnn_precision:.2f}")
    print(f"  Recall: {maskrcnn_recall:.2f}")
    print(f"  F1-score: {maskrcnn_f1:.2f}")
    print(f"  Accuracy: {maskrcnn_accuracy:.2f}")
    
    print(f"YOLO Metrics:")
    print(f"  Precision: {yolo_precision:.2f}")
    print(f"  Recall: {yolo_recall:.2f}")
    print(f"  F1-score: {yolo_f1:.2f}")
    print(f"  Accuracy: {yolo_accuracy:.2f}")

if __name__ == "__main__":
    main()

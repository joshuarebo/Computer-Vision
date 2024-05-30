# create_sample_video.py
import cv2
import numpy as np

def create_sample_video(output_path, width=640, height=480, fps=30, duration=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(int(fps * duration)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw some shapes to simulate objects
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.circle(frame, (400, 300), 50, (255, 0, 0), -1)  # Blue circle
        cv2.putText(frame, f'Frame {i+1}', (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f'Sample video saved to {output_path}')

if __name__ == "__main__":
    create_sample_video('sample_video.mp4')

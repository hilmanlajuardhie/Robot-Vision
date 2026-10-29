import cv2 as cv
import torch
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = WORKSPACE_ROOT / "model" / "yolo11m-pose.pt"
OUTPUT_DIR = WORKSPACE_ROOT / "runs" / "pose" / "hand"

def track_hands():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running YOLO11 on device: {device}")
    
    model = YOLO(str(MODEL_PATH))

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 60)

    dummy_input = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(
            source=dummy_input, 
            device=device, 
            verbose=False, 
            quantize=16
            )

    frame_count = 0
    fps_display = 0.0
    last_time = time.time()

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(
                source=frame, 
                device=device, 
                verbose=False,
                save=True,
                project=str(OUTPUT_DIR.parent),
                name=OUTPUT_DIR.name,
                exist_ok=True,
                )        
            
            annotated_frame = frame.copy()

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                
                keypoints = results[0].keypoints.data
                
                for person in keypoints:
                    if len(person) >= 11:
                        left_wrist = person[9]
                        right_wrist = person[10]

                        if left_wrist[2] > 0.5:
                            lx, ly = int(left_wrist[0]), int(left_wrist[1])
                            cv.circle(annotated_frame, (lx, ly), 12, (255, 0, 0), -1)  # Blue circle
                            cv.putText(annotated_frame, "Left Hand", (lx - 40, ly - 20), 
                                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

                        if right_wrist[2] > 0.5:
                            rx, ry = int(right_wrist[0]), int(right_wrist[1])
                            cv.circle(annotated_frame, (rx, ry), 12, (0, 0, 255), -1)  # Red circle
                            cv.putText(annotated_frame, "Right Hand", (rx - 40, ry - 20), 
                                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - last_time

            if elapsed_time >= 1.0:
                fps_display = frame_count / elapsed_time
                frame_count = 0
                last_time = current_time
                print(f"FPS: {fps_display:.2f}")

            fps_text = f"FPS: {fps_display:.2f}"
            cv.putText(annotated_frame, fps_text, (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (235,150,0), 2, cv.LINE_AA)

            annotated_frame = cv.flip(annotated_frame, 1)
            cv.imshow("YOLO11 Hand Tracking", annotated_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    track_hands()

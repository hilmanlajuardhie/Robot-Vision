import os
import time
import cv2 as cv
import numpy as np
from ultralytics import YOLO

# 1. Station Polygon Coordinates
STATIONS = {
    "Station_1": np.array([[10, 25], [110, 25], [110, 125], [10, 125]], np.int32),
    "Station_2": np.array([[10, 150], [110, 150], [110, 250], [10, 250]], np.int32),
    "Station_3": np.array([[270, 75], [370, 75], [370, 175], [270, 175]], np.int32),
    "Station_4": np.array([[490, 25], [630, 25], [630, 125], [490, 125]], np.int32),
    "Station_5": np.array([[490, 150], [630, 150], [630, 250], [490, 250]], np.int32),
}

# Tracking & On-Screen Display Memory
wrist_data = {}
station_last_time = {}  # Stores last completed time for each station: {"Station_1": 4.68}
log_history = []        # Stores up to 4 recent log strings for the on-screen box

def add_log(message):
    """Adds timestamped log message to on-screen queue."""
    timestamp = time.strftime("%H:%M:%S")
    log_history.append(f"[{timestamp}] {message}")
    if len(log_history) > 4:
        log_history.pop(0)

def draw_log_panel(frame):
    """Draws a semi-transparent log box in the bottom-center of the screen."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Position panel at bottom center
    p_x1, p_y1 = (w // 2) - 200, h - 110
    p_x2, p_y2 = (w // 2) + 200, h - 10
    
    # Dark semi-transparent background
    cv.rectangle(overlay, (p_x1, p_y1), (p_x2, p_y2), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Panel Header
    cv.putText(frame, "--- ON-SCREEN LOG SYSTEM ---", (p_x1 + 15, p_y1 + 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)
    
    # Render Log Lines
    for idx, log_text in enumerate(log_history):
        y_pos = p_y1 + 36 + (idx * 16)
        cv.putText(frame, log_text, (p_x1 + 15, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv.LINE_AA)

def track_wrist_stations():
    
    model_path = os.path.join("model", "yolo11m-pose.pt")
    model = YOLO(model_path)
    cap = cv.VideoCapture(0, cv.CAP_V4L2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        current_time = time.time()

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        # 1. Draw Station Polygons with Last Work Time
        for st_name, polygon in STATIONS.items():
            cv.polylines(frame, [polygon], isClosed=True, color=(255, 200, 0), thickness=2)
            
            # Display station name + last completed duration
            label = st_name
            if st_name in station_last_time:
                label += f" (Last: {station_last_time[st_name]:.2f}s)"
                
            cv.putText(frame, label, (polygon[0][0], polygon[0][1] - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # 2. Extract Keypoints & State Tracking
        if results[0].keypoints is not None and results[0].boxes.id is not None:
            keypoints_xy = results[0].keypoints.xy.cpu().numpy()
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for kpts, confs, track_id in zip(keypoints_xy, keypoints_conf, track_ids):
                wrists = [("Left_Wrist", 9), ("Right_Wrist", 10)]

                for wrist_label, kp_idx in wrists:
                    conf = confs[kp_idx] if confs is not None else 1.0
                    if conf < 0.5:
                        continue

                    wx, wy = int(kpts[kp_idx][0]), int(kpts[kp_idx][1])
                    if wx == 0 and wy == 0:
                        continue

                    wrist_id = f"Person_{track_id}_{wrist_label}"

                    # Polygon Check
                    active_station = None
                    for st_name, polygon in STATIONS.items():
                        if cv.pointPolygonTest(polygon, (wx, wy), False) >= 0:
                            active_station = st_name
                            break

                    if wrist_id not in wrist_data:
                        wrist_data[wrist_id] = {
                            "current_station": None,
                            "entry_time": None
                        }

                    obj = wrist_data[wrist_id]

                    # 3. Timer State Machine & On-Screen Log Trigger
                    if active_station != obj["current_station"]:
                        # EXIT EVENT: Calculate total time and print to screen log
                        if obj["current_station"] is not None:
                            elapsed = current_time - obj["entry_time"]
                            log_msg = f"{wrist_id} left {obj['current_station']}: {elapsed:.2f}s"
                            
                            # Print to terminal
                            print(f"[LOG] {log_msg}")
                            
                            # Save last completed time for the station label
                            station_last_time[obj["current_station"]] = elapsed
                            
                            # Send message to on-screen overlay panel
                            add_log(log_msg)

                        # ENTRY EVENT: Log station entry
                        if active_station is not None:
                            add_log(f"{wrist_id} entered {active_station}")

                        obj["current_station"] = active_station
                        obj["entry_time"] = current_time if active_station else None

                    # Active Duration
                    duration = 0.0
                    if obj["current_station"] and obj["entry_time"]:
                        duration = current_time - obj["entry_time"]

                    # Draw Wrist Marker
                    color = (0, 255, 0) if active_station else (0, 0, 255)
                    cv.circle(frame, (wx, wy), 6, color, -1)

                    text = f"{wrist_id}"
                    if active_station:
                        text += f" | {active_station}: {duration:.1f}s"

                    cv.putText(frame, text, (wx + 8, wy - 8),
                               cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 4. Render On-Screen Log Box
        draw_log_panel(frame)

        cv.imshow("Wrist Station ROI & Timer Check", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    track_wrist_stations()
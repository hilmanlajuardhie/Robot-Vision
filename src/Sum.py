"""Standalone interactive collection of all Robot-Vision source functions."""

import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = WORKSPACE_ROOT / "asset" / "img"
MODEL_DIR = WORKSPACE_ROOT / "model"


def _camera():
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def deteksi():
    """Live YOLOv11 object detection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running YOLO11 on device: {device}")
    model = YOLO(str(MODEL_DIR / "yolo11m.pt"))
    cap = _camera()
    if cap is None:
        return
    try:
        with torch.inference_mode():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = model.predict(frame, device=device, verbose=False)[0]
                cv.imshow("YOLO11 Detection", result.plot())
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv.destroyAllWindows()


def track_hands():
    """Live YOLOv11 pose wrist tracking."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running YOLO11 on device: {device}")
    model = YOLO(str(MODEL_DIR / "yolo11m-pose.pt"))
    cap = _camera()
    if cap is None:
        return
    fps, count, last = 0.0, 0, time.time()
    try:
        with torch.inference_mode():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv.flip(frame, 1)
                result = model.predict(frame, device=device, verbose=False)[0]
                output = frame.copy()
                if result.keypoints is not None and len(result.keypoints.data) > 0:
                    for person in result.keypoints.data:
                        if len(person) < 11:
                            continue
                        for name, index, color in (("Left Hand", 10, (255, 0, 0)),
                                                   ("Right Hand", 9, (0, 0, 255))):
                            point = person[index]
                            if point[2] > 0.5:
                                x, y = int(point[0]), int(point[1])
                                cv.circle(output, (x, y), 12, color, -1)
                                cv.putText(output, name, (x - 40, y - 20),
                                           cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
                count += 1
                now = time.time()
                if now - last >= 1:
                    fps, count, last = count / (now - last), 0, now
                cv.putText(output, f"FPS: {fps:.2f}", (10, 30),
                           cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (235, 150, 0), 2)
                cv.imshow("YOLO11 Hand Tracking", output)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv.destroyAllWindows()


STATIONS = {
    "Station_1": np.array([[10, 25], [160, 25], [160, 125], [10, 125]], np.int32),
    "Station_2": np.array([[10, 250], [160, 250], [160, 350], [10, 350]], np.int32),
    "Station_3": np.array([[245, 25], [395, 25], [395, 125], [245, 125]], np.int32),
    "Station_4": np.array([[480, 25], [630, 25], [630, 125], [480, 125]], np.int32),
    "Station_5": np.array([[480, 250], [630, 250], [630, 350], [480, 350]], np.int32),
}
wrist_data, station_last_time, log_history = {}, {}, []


def add_log(message):
    log_history.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    del log_history[:-4]


def draw_log_panel(frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w // 2 - 200, h - 110, w // 2 + 200, h - 10
    overlay = frame.copy()
    cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv.putText(frame, "--- ON-SCREEN LOG SYSTEM ---", (x1 + 15, y1 + 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)
    for i, message in enumerate(log_history):
        cv.putText(frame, message, (x1 + 15, y1 + 36 + i * 16),
                   cv.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv.LINE_AA)


def track_wrist_stations():
    """Live pose tracking with wrist station timers."""
    model = YOLO(str(MODEL_DIR / "yolo11m-pose.pt"))
    cap = _camera()
    if cap is None:
        return
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            now = time.time()
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            for name, polygon in STATIONS.items():
                cv.polylines(frame, [polygon], True, (0, 0, 153), 2)
                x = int(polygon[0][0])
                cv.putText(frame, name, (x, int(polygon[:, 1].min()) - 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 153), 1)
                if name in station_last_time:
                    cv.putText(frame, f"Last: {station_last_time[name]:.2f}s",
                               (x, int(polygon[:, 1].max()) + 18),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 153), 1)
            if result.keypoints is not None and result.boxes.id is not None:
                points = result.keypoints.xy.cpu().numpy()
                scores = result.keypoints.conf.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                for person, confidence, track_id in zip(points, scores, ids):
                    for label, index in (("Left_Wrist", 10), ("Right_Wrist", 9)):
                        if confidence[index] < 0.5:
                            continue
                        wx, wy = int(person[index][0]), int(person[index][1])
                        active = next((n for n, p in STATIONS.items()
                                       if cv.pointPolygonTest(p, (wx, wy), False) >= 0), None)
                        key = f"Person_{track_id}_{label}"
                        state = wrist_data.setdefault(key, {"current_station": None, "entry_time": None})
                        if active != state["current_station"]:
                            if state["current_station"] is not None:
                                elapsed = now - state["entry_time"]
                                old = state["current_station"]
                                station_last_time[old] = elapsed
                                message = f"{key} left {old}: {elapsed:.2f}s"
                                print(f"[LOG] {message}")
                                add_log(message)
                            if active:
                                add_log(f"{key} entered {active}")
                            state["current_station"] = active
                            state["entry_time"] = now if active else None
                        duration = now - state["entry_time"] if state["entry_time"] else 0
                        color = (0, 255, 0) if active else (0, 0, 255)
                        cv.circle(frame, (wx, wy), 6, color, -1)
                        text = key + (f" | {active}: {duration:.1f}s" if active else "")
                        cv.putText(frame, text, (wx + 8, wy - 8),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            draw_log_panel(frame)
            cv.imshow("Wrist Station ROI & Timer Check", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def estimate_aruco_pose():
    """Live ArUco marker detection and pose estimation."""
    cap = _camera()
    if cap is None:
        return
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    detector = cv.aruco.ArucoDetector(dictionary, cv.aruco.DetectorParameters())
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], np.float32)
    dist_coeffs = np.zeros((4, 1))
    size, half = 0.10, 0.05
    object_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], np.float32)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            corners, ids, _ = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            if ids is not None:
                cv.aruco.drawDetectedMarkers(frame, corners, ids)
                for marker in corners:
                    ok, rvec, tvec = cv.solvePnP(object_points, marker[0], camera_matrix, dist_coeffs)
                    if ok:
                        cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, size / 2)
                        x, y = map(int, marker[0][0])
                        cv.putText(frame, f"Dist: {tvec[2][0]:.2f}m", (x, y - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.imshow("Pose Estimation", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def img():
    """Basic OpenCV image manipulation exercise."""
    image = cv.imread(str(ASSET_DIR / "1.jpg"), 1)
    if image is None:
        print(f"Could not load {ASSET_DIR / '1.jpg'}")
        return
    for i in range(200):
        for j in range(image.shape[1]):
            image[i][j] = [random.randint(0, 255) for _ in range(3)]
    print(image.shape)
    tag = image[125:225, 275:375]
    image[350:450, 550:650] = tag
    cv.imshow("Showing", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def video():
    """Basic OpenCV four-panel video exercise."""
    cap = _camera()
    if cap is None:
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            height, width = frame.shape[:2]
            image = np.zeros(frame.shape, np.uint8)
            smaller = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
            image[:height // 2, :width // 2] = cv.rotate(smaller, cv.ROTATE_180)
            image[height // 2:, :width // 2] = smaller
            image[:height // 2, width // 2:] = cv.rotate(smaller, cv.ROTATE_180)
            image[height // 2:, width // 2:] = smaller
            cv.imshow("Showing", image)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def advanced_opencv():
    """Placeholder for the currently empty OpenCV_Advance.py file."""
    print("OpenCV_Advance.py currently has no implemented functions.")


PROGRAMS = {
    "1": ("YOLO object detection", deteksi),
    "2": ("YOLO hand/wrist tracking", track_hands),
    "3": ("ROI station tracking", track_wrist_stations),
    "4": ("ArUco pose estimation", estimate_aruco_pose),
    "5": ("OpenCV image exercise", img),
    "6": ("OpenCV video exercise", video),
    "7": ("Advanced OpenCV placeholder", advanced_opencv),
}


def interactive_menu():
    """Show a menu and run the selected function."""
    while True:
        print("\n=== Robot-Vision Sum.py ===")
        for key, (name, _) in PROGRAMS.items():
            print(f"{key}. {name}")
        print("q. Quit")
        choice = input("Select a function: ").strip().lower()
        if choice == "q":
            print("Goodbye.")
            return
        if choice not in PROGRAMS:
            print("Invalid selection.")
            continue
        print(f"Starting {PROGRAMS[choice][0]}. Press q in the OpenCV window to return.")
        try:
            PROGRAMS[choice][1]()
        except KeyboardInterrupt:
            print("\nOperation interrupted.")
            cv.destroyAllWindows()


if __name__ == "__main__":
    interactive_menu()

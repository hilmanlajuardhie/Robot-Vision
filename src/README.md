# `src/` — Robot-Vision programs

This directory contains the executable computer-vision programs for the workspace. Run commands from the workspace root:

```bash
python3 src/<filename>.py
```

## Programs

### `obj-detect.py`

`deteksi()` opens webcam `0` and performs live YOLOv11 object detection using `model/yolo11m.pt`. The annotated video is displayed in an OpenCV window. Press `q` to quit.

### `Hand_Tracking.py`

`track_hands()` performs live YOLOv11 pose inference using `model/yolo11m-pose.pt`. It identifies the left and right wrist keypoints, draws labels, and displays FPS. The camera image is mirrored before inference so the annotations remain readable. Press `q` to quit.

### `ROI_Detect.py`

`track_wrist_stations()` uses YOLO pose tracking with ByteTrack to monitor wrist movement through five predefined station polygons. It records entry and exit events, measures station time, and displays an on-screen log. The polygons are defined in the `STATIONS` dictionary and currently assume a 640×480 frame. Press `q` to quit.

### `ArUco.py`

`estimate_aruco_pose()` detects `DICT_6X6_250` ArUco markers and estimates their pose and distance using `solvePnP`. The camera calibration values and marker size are configured in the file. Press `q` to quit.

### `OpenCV_Basic.py`

Provides two basic exercises:

- `img()` loads `asset/img/1.jpg`, applies pixel manipulation, and displays the result.
- `video()` creates a four-panel webcam display using image resizing and rotation.

When run directly, it executes both exercises in sequence.

### `OpenCV_Advance.py`

Reserved for future advanced OpenCV experiments. It currently contains no executable functionality.

### `Sum.py`

`Sum.py` is the unified launcher. It reuses the functions from the other source files and provides one command-line entry point:

```bash
python3 src/Sum.py --list
python3 src/Sum.py object
python3 src/Sum.py hand
python3 src/Sum.py roi
python3 src/Sum.py aruco
python3 src/Sum.py image
python3 src/Sum.py video
```

The original files remain independently runnable. `Sum.py` does not run programs automatically when imported.

## Shared conventions

- Model files are loaded from the workspace-level `model/` directory.
- Image assets are loaded from the workspace-level `asset/img/` directory.
- Paths are resolved relative to each script's location rather than the shell's current directory.
- Live programs use webcam index `0` and exit with `q`.
- YOLO programs require PyTorch, Ultralytics, OpenCV, and NumPy.

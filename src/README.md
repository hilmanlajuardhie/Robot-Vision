# `src/` Source Code Guide

The `src/` directory contains the executable vision programs for the Robot-Vision workspace. The programs use OpenCV for camera access and display, Ultralytics YOLOv11 for detection and pose estimation, and NumPy for image and geometry operations.

All programs are designed to be run from the workspace root:

```bash
cd /home/alpha/DEV/Robot-Vision
python3 src/<program>.py
```

The scripts also resolve model and asset paths relative to the workspace location, so they do not depend on the terminal's current directory.

## Source files at a glance

| File | Main function | Purpose |
|---|---|---|
| `obj-detect.py` | `deteksi()` | Live YOLO object detection |
| `Hand_Tracking.py` | `track_hands()` | Live wrist tracking with YOLO pose |
| `ROI_Detect.py` | `track_wrist_stations()` | Wrist station/ROI timers with ByteTrack |
| `ArUco.py` | `estimate_aruco_pose()` | ArUco marker pose and distance estimation |
| `OpenCV_Basic.py` | `img()`, `video()` | Basic image and webcam exercises |
| `OpenCV_Advance.py` | None currently | Reserved for future OpenCV work |
| `Sum.py` | `interactive_menu()` | Standalone combined interactive program |

## File details

### `obj-detect.py`

Function: `deteksi()`

This program performs live object detection from webcam `0`.

Processing sequence:

1. Selects CUDA when `torch.cuda.is_available()` is true; otherwise uses CPU.
2. Loads `model/yolo11m.pt`.
3. Opens the camera through the V4L2 backend.
4. Captures frames continuously.
5. Runs YOLO prediction on each frame.
6. Draws the detected objects and confidence labels.
7. Displays the result in the `YOLO11 Detection` window.

Press `q` in the OpenCV window to stop. The model is the medium YOLO detection model, so it generally provides better accuracy than the nano model at the cost of inference speed.

### `Hand_Tracking.py`

Function: `track_hands()`

This program uses `model/yolo11m-pose.pt` to identify human pose keypoints and mark the wrists. It is a wrist tracker based on a human-pose model, not a dedicated finger or hand-landmark model.

Processing sequence:

1. Selects CUDA or CPU.
2. Loads the YOLO pose model.
3. Opens webcam `0` at a target resolution of 640×480.
4. Mirrors each frame before inference.
5. Reads keypoints for the left and right wrists.
6. Draws colored wrist circles and labels.
7. Displays the calculated FPS.

The pose keypoint indices used by the script are:

```text
Index 9  - right wrist in the YOLO pose keypoint layout
Index 10 - left wrist in the YOLO pose keypoint layout
```

Press `q` to stop. The camera mirror is applied before annotations so the text remains readable. If prediction output is configured for saving, generated frames are stored under `runs/pose/hand/`.

### `ROI_Detect.py`

Main function: `track_wrist_stations()`

Helper functions:

- `add_log(message)` keeps the most recent station events.
- `draw_log_panel(frame)` renders the on-screen event panel.

This program combines YOLO pose estimation, ByteTrack identity tracking, polygon tests, and a timer state machine.

The five work areas are defined in `STATIONS`:

```python
STATIONS = {
    "Station_1": ...,
    "Station_2": ...,
    "Station_3": ...,
    "Station_4": ...,
    "Station_5": ...,
}
```

Each station is a polygon described by pixel coordinates. The current coordinates are designed for a 640×480 frame. If the camera resolution or physical arrangement changes, edit these points.

Runtime behavior:

1. Opens webcam `0` and mirrors the frame.
2. Runs YOLO pose tracking with `persist=True` and `bytetrack.yaml`.
3. Extracts each tracked person's wrist keypoints and track ID.
4. Tests whether each wrist is inside a station polygon.
5. Starts a timer on station entry.
6. Stops the timer when the wrist leaves or enters another station.
7. Displays the current station, elapsed time, and last completed time.
8. Writes exit events to the terminal and the on-screen log panel.

The dictionaries `wrist_data`, `station_last_time`, and `log_history` store the runtime state. Press `q` to stop.

### `ArUco.py`

Function: `estimate_aruco_pose()`

This program detects markers from the OpenCV `DICT_6X6_250` dictionary and estimates their 3D pose.

Important configuration values:

- `camera_matrix` — intrinsic camera calibration matrix.
- `dist_coeffs` — lens distortion coefficients.
- `marker_size_meters` — physical marker size, currently 0.10 m.
- `obj_points` — 3D corner coordinates of the marker.

The displayed distance is the Z component of the translation vector returned by `solvePnP`. The built-in camera matrix is an example calibration; accurate distance measurements require calibration for the actual camera. Press `q` to stop.

### `OpenCV_Basic.py`

Function: `img()`

- Loads `asset/img/1.jpg`.
- Modifies pixels in a selected region with random colors.
- Copies a rectangular image region to another position.
- Displays the result until a key is pressed.

Function: `video()`

- Opens webcam `0`.
- Resizes each frame to half size.
- Places rotated and unrotated copies into a four-panel layout.
- Displays the layout until `q` is pressed.

When this file is run directly, it calls `img()` first and then `video()`.

### `OpenCV_Advance.py`

This file is currently a placeholder. It has no implemented functions or camera behavior yet.

### `Sum.py`

`Sum.py` is a standalone combined source file. It contains its own implementations of the functionality from the other source files rather than importing them.

Functions included:

- `deteksi()` — YOLO object detection.
- `track_hands()` — YOLO wrist tracking.
- `track_wrist_stations()` — ROI station tracking.
- `add_log()` — ROI event log storage.
- `draw_log_panel()` — ROI event panel rendering.
- `estimate_aruco_pose()` — ArUco pose estimation.
- `img()` — OpenCV image exercise.
- `video()` — OpenCV video exercise.
- `advanced_opencv()` — placeholder for advanced OpenCV work.
- `interactive_menu()` — interactive function selector.

Run it with:

```bash
python3 src/Sum.py
```

The menu is:

```text
1. YOLO object detection
2. YOLO hand/wrist tracking
3. ROI station tracking
4. ArUco pose estimation
5. OpenCV image exercise
6. OpenCV video exercise
7. Advanced OpenCV placeholder
q. Quit
```

For live options, press `q` inside the OpenCV window to stop the selected operation. The program then returns to the menu. Use `Ctrl+C` in the terminal if the window is unavailable or unresponsive.

## Shared paths

The scripts use `pathlib.Path` and derive the workspace root from the source file:

```python
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = WORKSPACE_ROOT / "model"
ASSET_DIR = WORKSPACE_ROOT / "asset" / "img"
```

Models:

- `model/yolo11m.pt` — object detection.
- `model/yolo11m-pose.pt` — human pose and wrist tracking.

Assets:

- `asset/img/1.jpg` — OpenCV image exercise.
- `asset/img/bus.jpg` — sample detection image.
- `asset/img/aruco_mark_id_7.png` and `aruco_mark_id_73.png` — marker images.

## Requirements

Core packages:

```bash
python3 -m pip install numpy opencv-python torch ultralytics
```

For GPU inference, install a PyTorch build compatible with the installed NVIDIA driver and CUDA version. A GUI-capable OpenCV installation is required because the programs use `cv.imshow()`.

## Camera settings

Most live programs use:

- Camera index: `0`.
- Backend: `cv.CAP_V4L2`.
- Target frame size: 640×480.
- Quit key: `q`.

If the camera cannot be opened, check available devices:

```bash
ls /dev/video*
```

Then change the camera index in the relevant source function, for example:

```python
cap = cv.VideoCapture(1, cv.CAP_V4L2)
```

If V4L2 does not work on the system, try `cv.VideoCapture(0)` without the backend argument.

## Common issues

### Model not found

Confirm that the expected file exists:

```bash
ls -lh model/yolo11m.pt model/yolo11m-pose.pt
```

### CUDA is unavailable

The scripts automatically fall back to CPU. Inference will still work but may be slower. Run `python3 Sys_Check.py` to inspect CUDA availability.

### GStreamer or camera warnings

Live camera position warnings are often harmless, but an `Internal data stream error` usually means the camera backend, device index, or requested format is unsupported. Try another camera index or backend.

### OpenCV window does not appear

The environment needs a graphical desktop session and an OpenCV build with GUI support. Headless environments cannot display `cv.imshow()` windows without additional display configuration.

## Development notes

The individual files are useful for focused testing. `Sum.py` is useful for interactive demonstrations and for keeping a single self-contained reference of the available functions. When changing logic, update both the individual source file and its corresponding implementation in `Sum.py` if both versions should remain synchronized.

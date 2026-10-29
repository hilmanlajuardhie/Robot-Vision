# Robot Vision

Experimental computer-vision software for robotics, using Python, OpenCV, and Ultralytics YOLOv11. The workspace contains standalone webcam programs for object detection, human pose/wrist tracking, and ArUco marker pose estimation. It is prepared for future ROS 2 integration, but is not yet a ROS 2 package.

## Workspace layout

```text
Robot-Vision/
├── asset/img/       Input images and ArUco markers
├── model/            YOLOv11 model weights
├── src/              Vision scripts
├── runs/             Generated prediction images
├── Sys_Check.py      Dependency and hardware diagnostics
├── routine.sh        Optional Ubuntu/Python maintenance script
└── README.md
```

Source code resolves paths relative to the workspace root, so assets belong in `asset/` and model weights belong in `model/` regardless of the directory from which a script is launched.

## Models and assets

- `model/yolo11n.pt` — small object-detection model
- `model/yolo11m.pt` — medium object-detection model
- `model/yolo11m-pose.pt` — medium human-pose model
- `asset/img/bus.jpg` — detection test image
- `asset/img/aruco_mark_id_7.png` and `aruco_mark_id_73.png` — ArUco test markers

## Requirements

- Python 3.10+
- OpenCV with GUI support and a webcam for live scripts
- NumPy, PyTorch, and Ultralytics
- Optional: CUDA-capable PyTorch for GPU inference
- Optional diagnostics: Gymnasium, Stable-Baselines3, MuJoCo, ONNX Runtime, and GStreamer

Install the core packages with:

```bash
python3 -m pip install numpy opencv-python torch ultralytics
```

Choose the PyTorch package appropriate for your CUDA version when GPU acceleration is needed.

## Running the programs

Run from the workspace root, or use an absolute script path:

```bash
python3 src/obj-detect.py
```

Opens webcam `0`, runs YOLOv11 object detection with `model/yolo11m.pt`, and displays annotated video. Press `q` to quit.

```bash
python3 src/Hand_Tracking.py
```

Uses `model/yolo11m-pose.pt` to track human wrist keypoints, displays FPS, and saves output under `runs/pose/hand/`. This is pose-based wrist tracking, not a dedicated finger-landmark hand model.

```bash
python3 src/ArUco.py
```

Detects `DICT_6X6_250` ArUco markers, estimates pose with `solvePnP`, draws coordinate axes, and displays estimated Z distance. The camera matrix is an example 640×480 calibration; accurate measurements require calibration for the actual camera.

```bash
python3 src/ROI_Detect.py
```

Runs YOLOv11 pose tracking with ByteTrack and monitors five predefined rectangular work-station regions. The program tracks each detected person's left and right wrist, reports when a wrist enters or leaves a station, measures the time spent there, and displays the timers and recent event logs on screen. Press `q` to quit.

The station definitions are in the `STATIONS` dictionary near the top of `src/ROI_Detect.py`. Their coordinates currently assume a 640×480 camera image. Adjust these polygons if the camera resolution or physical station layout changes. The script uses `model/yolo11m-pose.pt` and webcam `0`.

`src/OpenCV_Basic.py` contains commented image and webcam exercises using `asset/img/`; `src/OpenCV_Advance.py` is currently a placeholder.

## Diagnostics

```bash
python3 Sys_Check.py
```

This checks Python packages, OpenCV GTK/GStreamer support, system GStreamer, PyTorch CUDA, Gymnasium/Stable-Baselines3, and MuJoCo rendering. Optional components may report warnings or failures when not installed.

## Output and current limitations

Generated results are stored in `runs/`. The ROI tracker currently displays its results live and logs station events to the terminal and on-screen panel. There are currently no ROS 2 nodes, launch files, `package.xml`, or `setup.py`. Camera index, resolution, model choice, ROI polygons, and ArUco calibration are configured in source code. A future ROS 2 version could publish detections, pose keypoints, station events, and marker transforms as ROS topics.

See [LICENSE](LICENSE) for licensing information.

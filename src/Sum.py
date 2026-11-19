"""Unified launcher for the Robot-Vision source programs.

Examples:
    python3 src/Sum.py object
    python3 src/Sum.py hand
    python3 src/Sum.py roi
    python3 src/Sum.py aruco
    python3 src/Sum.py image
    python3 src/Sum.py video
    python3 src/Sum.py --list

The original source files remain available for direct execution. This file
reuses their functions instead of maintaining duplicate implementations.
"""

import argparse
import importlib.util
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

from ArUco import estimate_aruco_pose
from Hand_Tracking import track_hands
from OpenCV_Basic import img, video
from ROI_Detect import track_wrist_stations


def _load_object_detection_function():
    """Load deteksi() from obj-detect.py, whose filename contains a hyphen."""
    module_path = SRC_DIR / "obj-detect.py"
    spec = importlib.util.spec_from_file_location("robot_vision_object_detection", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load object detection module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.deteksi


deteksi = _load_object_detection_function()

PROGRAMS = {
    "object": (deteksi, "YOLO object detection"),
    "hand": (track_hands, "YOLO wrist/hand tracking"),
    "roi": (track_wrist_stations, "Wrist station ROI tracking"),
    "aruco": (estimate_aruco_pose, "ArUco pose estimation"),
    "image": (img, "OpenCV image exercise"),
    "video": (video, "OpenCV video exercise"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run one of the Robot-Vision programs from a single entry point."
    )
    parser.add_argument(
        "program",
        nargs="?",
        choices=sorted(PROGRAMS),
        help="program to run (use --list to show descriptions)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available programs without starting the camera",
    )
    args = parser.parse_args()

    if args.list or args.program is None:
        print("Available Robot-Vision programs:")
        for name, (_, description) in PROGRAMS.items():
            print(f"  {name:8} {description}")
        if args.program is None:
            print("\nExample: python3 src/Sum.py roi")
        return

    function, description = PROGRAMS[args.program]
    print(f"Starting: {description}")
    function()


if __name__ == "__main__":
    main()

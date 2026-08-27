import torch
import cv2 as cv
import matplotlib as plt
import numpy as np
from ultralytics import YOLO

print(f"OpenCV Version: {cv.__version__}")
print(f"Matplotlib Version: {plt.__version__}")
print(f"Numpy Version: {np.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

model = YOLO("vision/yolo11n.pt")
result = model.predict(
    source="vision/asset/img/bus.jpg",
    device=0 if torch.cuda.is_available() else "cpu",
    save = True,
    name="Result"
)
print("✅ YOLOv11 executed successfully on GPU!")

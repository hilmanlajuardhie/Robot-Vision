import cv2 as cv
import numpy as np
import torch

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
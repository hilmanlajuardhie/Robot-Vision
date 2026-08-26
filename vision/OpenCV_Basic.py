import cv2 as cv
import time
import numpy as np
import random

# ========== Imaging ==========
# img = cv.imread('asset/4.jpg', 1)
# for i in range (200):
#     for j in range (img.shape[1]):
#         img[i][j] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]

# print(img.shape)
# tag = img[125:225, 275:375]
# img[350:450, 550:650] = tag

# cv.imshow('Showing',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# ========== Video ==========
# cap = cv.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     width = int(cap.get(3))
#     height = int(cap.get(4))

#     image = np.zeros(frame.shape, np.uint8)
#     smaller_frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5)

#     image[:height//2, :width//2] = cv.rotate(smaller_frame, cv.ROTATE_180)
#     image[height//2:, :width//2] = smaller_frame
#     image[:height//2, width//2:] = cv.rotate(smaller_frame, cv.ROTATE_180)
#     image[height//2:, width//2:] = smaller_frame

#     cv.imshow('Showing', image)
#     if cv.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()
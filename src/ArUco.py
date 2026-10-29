import cv2 as cv
import numpy as np
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = WORKSPACE_ROOT / "asset" / "img"

# ========== Generating ArUco Marker ==========
# def gen_aruco_marker():
#     dicti = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
#     mark_id = 73
#     mark_size = 600
#     mark_image = cv.aruco.generateImageMarker(dicti, mark_id, mark_size)

#     border_size = 10
#     mark_image_border = cv.copyMakeBorder(
#         mark_image,
#         border_size, border_size, border_size, border_size,
#         cv.BORDER_CONSTANT,
#         value=[255, 255, 255]
#     )

#     file_name = ASSET_DIR / f"aruco_mark_id_{mark_id}.png"
#     cv.imwrite(file_name, mark_image_border)

#     print(f"Marker Generated and Saved as {file_name}")

#     cv.imshow("Generated ArUco Marker", mark_image_border)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     gen_aruco_marker()

# ========== Detect ArUco Marker ==========
def estimate_aruco_pose():
    cap = cv.VideoCapture(0)

    prev_frame_time = 0
    new_frame_time = 0
    

    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    camera_matrix = np.array([
        [800.0,   0.0, 320.0],
        [  0.0, 800.0, 240.0],
        [  0.0,   0.0,   1.0]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion for this example

    marker_size_meters = 0.10 # e.g., 10 centimeters (0.1 meters)

    half_size = marker_size_meters / 2.0
    obj_points = np.array([
        [-half_size,  half_size, 0], # Top-left
        [ half_size,  half_size, 0], # Top-right
        [ half_size, -half_size, 0], # Bottom-right
        [-half_size, -half_size, 0]  # Bottom-left
    ], dtype=np.float32)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i in range(len(ids)):
                img_points = corners[i][0]
                
                success, rvec, tvec = cv.solvePnP(
                    obj_points, 
                    img_points, 
                    camera_matrix, 
                    dist_coeffs
                )
                
                if success:
                    cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size_meters / 2)
                    
                    distance_z = tvec[2][0]
                    
                    text_x = int(img_points[0][0])
                    text_y = int(img_points[0][1]) - 10
                    cv.putText(frame, f"Dist: {distance_z:.2f}m", (text_x, text_y), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow("Pose Estimation", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    estimate_aruco_pose()

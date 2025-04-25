import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import logging

logger = logging.getLogger(__name__)

# Checkerboard size (number of inner corners)
CHECKERBOARD = (10, 7)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3d points
imgpoints = []  # 2d points

images = glob.glob('./Calibration/macbooktheo/*.jpg')
logger.info("Calibration images retreived")

img_size = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        if img_size is None:
            print("img_size: None, ", img_size)
            img_size = gray.shape[::1]
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('Corners', img)
        # cv2.waitKey(100)
    else:
        logger.warning(f"No checkboard fond in image {fname}")

logger.info("Calibration images processed")

if img_size is None:
    logger.error("No valid image found for calibration.")
    exit()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

logger.info("Calibration done")
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)

camera_matrix = mtx
dist_coeffs = dist

# Set up ArUco dictionary and detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Estimate pose: 3D size of marker = 0.05m
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # Convert rotation vector to quaternion
            rot_matrix, _ = cv2.Rodrigues(rvec)
            quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
            position = tvec.flatten()  # [x, y, z]

            print(f"Position: {position}, Quaternion: {quat}")

    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

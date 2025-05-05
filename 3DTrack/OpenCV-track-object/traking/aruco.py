import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import logging

from calibration.capture import Calibrator

logger = logging.getLogger(__name__)


class Aruco3DTrack:
    camera_matrix = None
    dist_coeffs = None

    def __init__(camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def start_track():
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
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.camera_matrix, self.dist_coeffs)
                for rvec, tvec in zip(rvecs, tvecs):
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

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

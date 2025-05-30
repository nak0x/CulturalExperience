import cv2
import numpy as np
import glob
import logging

logger = logging.getLogger(__name__)


def calibrate(checkerboard = (10, 7)):
    # Checkerboard size (number of inner corners)
    CHECKERBOARD = checkerboard

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
    return {
        "matrix": mtx,
        "distCoeffs": dist,
        "retval": ret,
        "rvecs": rvecs,
        "tvecs": tvecs
    }

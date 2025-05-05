from calibration.calib import calibrate
from traking.aruco import Aruco3DTrack

calibration = calibrate()

tracker = Aruco3DTrack(calibration["matrix"], calibration["distCoeffs"])
tracker.start_track()

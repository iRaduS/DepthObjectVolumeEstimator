import os
from dotenv import load_dotenv
from CameraParametersDetection.CameraCalibration import CameraCalibration


load_dotenv()
DEBUG_MODE = os.getenv("DEBUG_MODE")

"""
Camera calibration with a defined dataset that can be found inside datasets/CameraParametersCalibration
"""
camera_calibration = CameraCalibration(DEBUG_MODE)
camera_calibration.extract_objective_calibration_from_images()
camera_calibration.extract_camera_matrix()

print(camera_calibration.cameraMatrix)

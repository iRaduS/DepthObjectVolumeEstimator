import os
import glob
import cv2 as cv
import numpy as np


def load_calibration_images_from_dataset():
    image_paths = glob.glob(os.path.join("./datasets/CameraParametersCalibration", "*.JPG"))
    if len(image_paths) == 0:
        raise Exception('Image directory of /datasets/CameraParametersCalibration couldn\'t find any .JPG files.')

    return np.array([
        cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2GRAY) for image_path in image_paths
    ])


def load_calibration_settings_from_env():
    return {
        "chessboard_shape": (7, 6),
        "picture_shape": (4032, 3024),
    }

import os
import glob
import cv2 as cv
import numpy as np


def load_calibration_images_from_dataset():
    image_paths = glob.glob(os.path.join("../datasets/CameraParametersCalibration", "*.png"))
    apply_transformation = lambda image: cv.cvtColor(image, cv.BGR2GRAY)

    images = np.array([cv.imread(image_path) for image_path in image_paths])
    return apply_transformation(images)


def load_calibration_settings_from_env():
    return {
        "chessboard_shape": (24, 17),
        "picture_shape": (1440, 1080),
    }

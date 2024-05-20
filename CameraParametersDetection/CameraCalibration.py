import cv2 as cv
import numpy as np
from tqdm import tqdm
from CameraParametersDetection import load_calibration_images_from_dataset, \
    load_calibration_settings_from_env


class CameraCalibration:
    def __init__(self, DEBUG_MODE):
        self.calibration_images = load_calibration_images_from_dataset()
        self.calibration_settings = load_calibration_settings_from_env()
        self.DEBUG_MODE = DEBUG_MODE
        self.object_coords = np.zeros((self.calibration_settings["chessboard_shape"][1] *
                                       self.calibration_settings["chessboard_shape"][0], 3), dtype=np.float32)
        self.object_coords[:, :2] = np.mgrid[0:self.calibration_settings["chessboard_shape"][0],
                                    0:self.calibration_settings["chessboard_shape"][1]].T.reshape(-1, 2)
        self.corner_coords = list()
        self.objective_points = list()
        self.cameraMatrix = None

    def current_image_per_size_string(self, i):
        return f'{i + 1}/{len(self.calibration_images)}'

    def extract_objective_calibration_from_images(self):
        chessboard_shape = self.calibration_settings["chessboard_shape"]
        for i, image in tqdm(enumerate(self.calibration_images)):
            result, corners = cv.findChessboardCorners(image, chessboard_shape, None)

            if result is not True:
                print(f'[CameraCalibration]: Couldn\'t find any chessboard on image '
                      f'with ID: {i} out [{self.current_image_per_size_string(i)}], '
                      f'[chessboard_shape: {chessboard_shape}], [image_shape: {image.shape}]')
                continue

            accurate_corners = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), (
                cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3
            ))
            self.corner_coords.append(accurate_corners)
            self.objective_points.append(self.object_coords)

            if self.DEBUG_MODE == "true":
                print(f'[CameraCalibration - DEBUG MODE]: Showing image {self.current_image_per_size_string(i)}')
                cv.drawChessboardCorners(image, chessboard_shape, accurate_corners, result)
                cv.imshow(f'Image {id} calibration chessboard', image)
                cv.waitKey(1000)
        cv.destroyAllWindows()

    def extract_camera_matrix(self):
        picture_shape = self.calibration_settings["picture_shape"]
        _, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(self.objective_points, self.corner_coords,
                                                                 picture_shape[::-1], None, None)

        self.cameraMatrix = cameraMatrix
        print(f'[CameraCalibration]: CameraMatrix was extracted with success, [value: {self.cameraMatrix}]')

        mean_error = 0.0
        for i, objective_point in tqdm(enumerate(self.objective_points)):
            object_points, _ = cv.projectPoints(objective_point, rvecs[i], tvecs[i], cameraMatrix, dist)
            mean_error += cv.norm(self.corner_coords[i], object_points, cv.NORM_L2) / len(object_points)

        print(f'[CameraCalibration]: Mean Error: {mean_error}/{len(self.corner_coords)}')

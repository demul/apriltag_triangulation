import glob
import os
import sys
import pickle

import cv2

from apriltag_detector import ApriltagDetector
from apriltag_visualizer import ApriltagVisualizer
from camera import Camera
from triangulator import Triangulator


CAMERA_PARAMETER_PATH = "data/camera_params.yaml"
CAMERA_DIR_ROOT_PATH = "data/"
IMAGE_NAME = "246.png"
DEPTH_FL_INFRA_PAIR = [1, 2]
DEPTH_FR_INFRA_PAIR = [3, 4]


if __name__ == "__main__":
    sys.stdin.flush()
    apriltag_detector = ApriltagDetector()
    apriltag_visualizer = ApriltagVisualizer()

    image_list = [cv2.imread(os.path.join(camera_dir_path, IMAGE_NAME)) 
        for camera_dir_path 
        in glob.glob(os.path.join(CAMERA_DIR_ROOT_PATH, "cam*"))]
    input_fd = cv2.FileStorage()
    input_fd.open(CAMERA_PARAMETER_PATH, cv2.FileStorage_READ)
    depth_fl_infra_left, depth_fl_infra_right= Camera.get_camera_pairs_from_yaml(
        input_fd,
        DEPTH_FL_INFRA_PAIR[0],
        DEPTH_FL_INFRA_PAIR[1])
    depth_fr_infra_left, depth_fr_infra_right= Camera.get_camera_pairs_from_yaml(
        input_fd,
        DEPTH_FR_INFRA_PAIR[0],
        DEPTH_FR_INFRA_PAIR[1])

    # depth_fl_triangulator = Triangulator(depth_fl_infra_left, depth_fl_infra_right)
    # depth_fr_triangulator = Triangulator(depth_fr_infra_left, depth_fr_infra_right)

    corner_list_of_marker_list_depth_fl_infra_left, id_list_depth_fl_infra_left = \
        apriltag_detector.detect_apriltag_corners(image_list[1])
    corner_list_of_marker_list_depth_fl_infra_right, id_list_depth_fl_infra_right = \
        apriltag_detector.detect_apriltag_corners(image_list[2])
    corner_list_of_marker_list_depth_fr_infra_left, id_list_depth_fr_infra_left = \
        apriltag_detector.detect_apriltag_corners(image_list[3])
    corner_list_of_marker_list_depth_fr_infra_right, id_list_depth_fr_infra_right = \
        apriltag_detector.detect_apriltag_corners(image_list[4])

    corners = [
        corner_list_of_marker_list_depth_fl_infra_left, 
        corner_list_of_marker_list_depth_fl_infra_right,
        corner_list_of_marker_list_depth_fr_infra_left, 
        corner_list_of_marker_list_depth_fr_infra_right,
        ]
    with open('corners.pickle', 'wb') as f:
        pickle.dump(corners, f, pickle.HIGHEST_PROTOCOL)

    # # cv2.imshow("", img_marked)
    # # while True:
    # #     key = cv2.waitKey(0)
    # #     if key == 27:
    # #         break

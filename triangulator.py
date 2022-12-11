import cv2
import numpy as np

from camera import Camera


class Triangulator:
    def __init__(self, infra_camera_left:Camera, infra_camera_right:Camera):
        self.infra_camera_left_new_intrinsic = None
        self.infra_camera_left_undistort_map_x, self.infra_camera_left_undistort_map_y = cv2.initUndistortRectifyMap(
            infra_camera_left.intrinsic,
            infra_camera_left.distortion,
            None,
            self.infra_camera_left_new_intrinsic,
            (infra_camera_left.width, infra_camera_left.height),
            cv2.CV_32FC2
            )
        self.infra_camera_left_projection_matrix = self.infra_camera_left_new_intrinsic @ infra_camera_left.rig_transformation[:3]

        self.infra_camera_right_new_intrinsic = None
        self.infra_camera_right_undistort_map_x, self.infra_camera_right_undistort_map_y = cv2.initUndistortRectifyMap(
            infra_camera_right.intrinsic,
            infra_camera_right.distortion,
            None,
            self.infra_camera_right_new_intrinsic,
            (infra_camera_right.width, infra_camera_right.height),
            cv2.CV_32FC2
            )
        self.infra_camera_right_projection_matrix = self.infra_camera_right_new_intrinsic @ infra_camera_right.rig_transformation[:3]

    def undistort_corners(corner_list_of_marker_list, undistortion_map_x, undistortion_map_y):
        for corner_list in corner_list_of_marker_list:
            for corner in corner_list:
                corner = np.array(undistortion_map_x[int(corner[0]), int(corner[1])],undistortion_map_y[int(corner[0]), int(corner[1])])
        return corner_list_of_marker_list

    def triangulate_corners(corner_list_of_marker_list0, corner_list_of_marker_list1):
        # TODO: 언디스토션-트라이앵귤레이션-시각화
        pass
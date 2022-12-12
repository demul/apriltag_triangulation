import cv2
import numpy as np

from camera import Camera


class Triangulator:
    def __init__(self, infra_camera_left: Camera, infra_camera_right: Camera):
        self.infra_camera_left = infra_camera_left
        self.infra_camera_left_forward_undistort_map = \
            Triangulator.get_forward_undistort_map(infra_camera_left, reproject=True)
        self.infra_camera_left_inverse_undistort_map = \
            Triangulator.get_inverse_undistort_map(infra_camera_left)
        # infra_camera_left_extrinsic = np.array([[1, 0, 0, 0],
        #                                         [0, 1, 0, 0],
        #                                         [0, 0, 1, 0]], dtype=np.float32)
        # self.infra_camera_left_projection_matrix = \
        #     self.infra_camera_left.intrinsic @ infra_camera_left_extrinsic
        self.infra_camera_left_projection_matrix = \
            self.infra_camera_left.intrinsic @ np.linalg.inv(infra_camera_left.rig_transformation)[:3]

        self.infra_camera_right = infra_camera_right
        self.infra_camera_right_forward_undistort_map = \
            Triangulator.get_forward_undistort_map(infra_camera_right, reproject=True)
        self.infra_camera_right_inverse_undistort_map = \
            Triangulator.get_inverse_undistort_map(infra_camera_right)
        # infra_camera_right_extrinsic = \
        #     Triangulator.get_right_to_left_transformation(infra_camera_left, infra_camera_right)[:3]
        # self.infra_camera_right_projection_matrix = \
        #     self.infra_camera_right.intrinsic @ infra_camera_right_extrinsic
        self.infra_camera_right_projection_matrix = \
            self.infra_camera_right.intrinsic @ np.linalg.inv(infra_camera_right.rig_transformation)[:3]

    @staticmethod
    def get_forward_undistort_map(camera, reproject=True):
        width = camera.width
        height = camera.height
        index_map = \
            np.transpose(
                np.indices([height, width], dtype=np.float32),
                [1, 2, 0])[:, :, ::-1].reshape(height * width, 2)
        index_map_undistorted = cv2.undistortPoints(
            index_map,
            camera.intrinsic,
            camera.distortion,
            None,
            camera.intrinsic if reproject else None
        )
        return index_map_undistorted.reshape(height, width, 2)

    @staticmethod
    def get_inverse_undistort_map(camera):
        return cv2.initUndistortRectifyMap(
            camera.intrinsic,
            camera.distortion,
            None,
            camera.intrinsic,
            (camera.width, camera.height),
            cv2.CV_32FC2)[0]

    @staticmethod
    def undistort_corners(corner_list_of_marker_list, undistortion_map):
        new_corner_list_of_marker_list = []
        for corner_list in corner_list_of_marker_list:
            new_corner_list = []
            for corner in corner_list:
                new_corner_list.append(np.array([undistortion_map[:, :, 0][round(corner[1]), round(corner[0])],
                                                undistortion_map[:, :, 1][round(corner[1]), round(corner[0])]]))
            new_corner_list_of_marker_list.append(new_corner_list)
        new_corner_list_of_marker_list = np.array(new_corner_list_of_marker_list, dtype=np.float32)
        return new_corner_list_of_marker_list

    def triangulate_corners(self,
                            corner_list_of_marker_list_and_id_list_left,
                            corner_list_of_marker_list_and_id_list_right):
        corner_list_of_marker_list_left, id_list_left = corner_list_of_marker_list_and_id_list_left
        corner_list_of_marker_list_right, id_list_right = corner_list_of_marker_list_and_id_list_right
        id_index_dict_left = dict((id_, index) for index, id_ in enumerate(id_list_left))
        id_index_dict_right = dict((id_, index) for index, id_ in enumerate(id_list_right))
        id_intersection = list(set(id_list_left).intersection(id_index_dict_right))
        intersected_corner_list_of_marker_list_left = \
            np.array([corner_list_of_marker_list_left[id_index_dict_left[id_]] for id_ in id_intersection],
                     dtype=np.float32)
        intersected_corner_list_of_marker_list_right = \
            np.array([corner_list_of_marker_list_right[id_index_dict_right[id_]] for id_ in id_intersection],
                     dtype=np.float32)

        intersected_corner_list_of_marker_list_left_undistorted = \
            cv2.undistortPoints(
                intersected_corner_list_of_marker_list_left.reshape(-1, 2),
                self.infra_camera_left.intrinsic,
                self.infra_camera_left.distortion,
                None,
                self.infra_camera_left.intrinsic
            ).reshape(-1, 4, 2)
        intersected_corner_list_of_marker_list_right_undistorted = \
            cv2.undistortPoints(
                intersected_corner_list_of_marker_list_right.reshape(-1, 2),
                self.infra_camera_right.intrinsic,
                self.infra_camera_right.distortion,
                None,
                self.infra_camera_right.intrinsic
            ).reshape(-1, 4, 2)

        # intersected_corner_list_of_marker_list_left_undistorted = \
        #     Triangulator.undistort_corners(
        #         intersected_corner_list_of_marker_list_left,
        #         self.infra_camera_left_forward_undistort_map
        #     )
        # intersected_corner_list_of_marker_list_right_undistorted = \
        #     Triangulator.undistort_corners(
        #         intersected_corner_list_of_marker_list_right,
        #         self.infra_camera_right_forward_undistort_map
        #     )

        triangulated_3d_points = cv2.triangulatePoints(
                self.infra_camera_left_projection_matrix,
                self.infra_camera_right_projection_matrix,
                intersected_corner_list_of_marker_list_left_undistorted.reshape(-1, 1, 2),
                intersected_corner_list_of_marker_list_right_undistorted.reshape(-1, 1, 2)
            ).T
        # return triangulated_3d_points[:, :3]
        return triangulated_3d_points[:, :3] / triangulated_3d_points[:, [3]]

    @staticmethod
    def get_right_to_left_transformation(left_camera, right_camera):
        transformation_right_to_left = (
                np.linalg.inv(right_camera.rig_transformation)
                @ left_camera.rig_transformation
        )
        return transformation_right_to_left

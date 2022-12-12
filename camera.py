import numpy as np


class Camera:
    def __init__(
        self,
        camera_name: str,
        width: int = 1280,
        height: int = 800,
        intrinsic: np.ndarray = None,
        distortion: np.ndarray = None,
        rig_transformation : np.ndarray = None,
    ):
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.rig_transformation = rig_transformation

    @staticmethod
    def read_camera_from_yaml(fd, camera_index):
        intrinsic_node = fd.getNode("cam%d_parameters" % camera_index)
        intrinsic = None
        if not intrinsic_node.empty():
            fx = intrinsic_node.getNode("fx").real()
            fy = intrinsic_node.getNode("fy").real()
            cx = intrinsic_node.getNode("cx").real()
            cy = intrinsic_node.getNode("cy").real()

            intrinsic = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]]
                                 , dtype=np.float64)

        distortion_node = fd.getNode("cam%d_distortions" % camera_index)
        distortion = None
        if not distortion_node.empty():
            distortion = distortion_node.mat()

        rotation_node = fd.getNode("cam%dRotation" % camera_index)
        rotation = None
        if not rotation_node.empty():
            rotation = rotation_node.mat()

        translation_node = fd.getNode("cam%dTranslation" % camera_index)
        translation = None
        if not translation_node.empty():
            translation = translation_node.mat()

        rig_transformation = np.zeros([4, 4], dtype=np.float64)
        rig_transformation[:3, :3] = rotation
        rig_transformation[:3, [3]] = translation
        rig_transformation[3, 3] = 1
        camera = Camera(camera_name=None, intrinsic=intrinsic, distortion=distortion, rig_transformation=rig_transformation)
        return camera

    @staticmethod
    def get_camera_pairs_from_yaml(fd, left_camera_index, right_camera_index):
        left_camera = Camera.read_camera_from_yaml(fd, left_camera_index)
        right_camera = Camera.read_camera_from_yaml(fd, right_camera_index)
        return left_camera, right_camera

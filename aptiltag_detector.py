from copy import copy

from apriltags_eth import make_default_detector
import cv2
import numpy as np
import yaml


class ApriltagDetector:
    def __init__(self):
        self.detector = make_default_detector()

    def detect_apriltag_corners(self, image):
        # image = cv2.GaussianBlur(image, (0, 0), 1.0)
        tag_list = self.detector.extract_tags(image)
        if tag_list:
            corner_list_of_tag_list = np.array(
                [np.array(tag.corners) for tag in tag_list]
            )
            id_list = [tag.id for tag in tag_list]
        else:
            corner_list_of_tag_list = []
            id_list = []
        return corner_list_of_tag_list, id_list


if __name__ == "__main__":
    apriltag_detector = ApriltagDetector()
    print("Finished!")

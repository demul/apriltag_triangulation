from copy import copy

import cv2
import numpy as np


class ApriltagVisualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.marker_color = (0, 0, 255)
        self.marker_radius = 2
        self.marker_thickness = 2

        self.marker_history_color = (255, 0, 0)
        self.marker_history_radius = 2
        self.marker_history_thickness = 2

        self.font_color = (0, 255, 0)
        self.font_size = 1.5
        self.font_thickness = 2

    def visualize_corners(self, image, corner_list_of_marker_list, id_list):
        image_copy = copy(image)
        corner_list_of_marker_list_int = corner_list_of_marker_list.astype(np.uint32)
        for corner_list, marker_id in zip(corner_list_of_marker_list_int, id_list):
            for corner in corner_list:
                image_copy = cv2.circle(
                    image_copy,
                    corner,
                    self.marker_radius,
                    self.marker_color,
                    self.marker_thickness,
                )
            cv2.putText(
                image_copy,
                str(marker_id),
                (corner_list[0, 0], corner_list[0, 1]),
                self.font,
                self.font_size,
                self.font_color,
                self.font_thickness,
                cv2.LINE_AA,
            )
        return image_copy
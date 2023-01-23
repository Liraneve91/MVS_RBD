# The detector based on the code here:
# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
import numpy as np
import cv2
import matplotlib
import SharedCode.PY.utilsScript as utils


class ColorsDetector:
    """
    supports finding one object ONLY
    """

    def __init__(self, boundaries_bgr: dict, float_pre: float):
        """
        :param boundaries_bgr:dict - 'color':{'low': bgr bottom range, 'high': bgr top range}
        :param float_pre:
        """
        self.boundaries_bgr = boundaries_bgr
        self.float_pre = float_pre

    def __str__(self):
        string = 'ColorsDetector:\n'
        string += '\tcolors_boundaries={}\n'.format(self.boundaries_bgr)
        string += '\tfloat_pre={}\n'.format(self.float_pre)
        return string

    def detect_colors(self, frame: np.array, get_drawing_dict: bool = False) -> (dict, dict):
        """
        :param frame:
        :param get_drawing_dict:
        looking for the best match rect over the color (between defined ranges) for each color on boundaries_rgb
        :return:
        """
        data_per_cam = {}
        for_drawing_dict = {"rects": {}} if get_drawing_dict else None
        # loop over the boundaries
        for color_name, high_low_dict in self.boundaries_bgr.items():
            # create NumPy arrays from the boundaries
            lower = np.array(high_low_dict['low'], dtype="float")
            upper = np.array(high_low_dict['high'], dtype="float")
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(frame, lower, upper)
            mask = mask.T
            mask_non_zero = np.nonzero(mask)
            if len(mask_non_zero[0]) == 0 or len(mask_non_zero[1]) == 0:
                continue

            min_x = np.min(mask_non_zero[0])
            max_x = np.max(mask_non_zero[0])
            min_y = np.min(mask_non_zero[1])
            max_y = np.max(mask_non_zero[1])
            if min_x == max_x or min_y == max_y:
                continue
            print("min_x: ", min_x, ", max_x: ", max_x, "min_y: ", min_y, "max_y: ", max_y)
            if get_drawing_dict:
                for_drawing_dict["rects"][color_name] = {
                    "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y
                }
            kp = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2], dtype=float)
            kp = np.round(kp.tolist(), self.float_pre).tolist()
            c = matplotlib.colors.to_rgba(color_name, alpha=None)
            c = np.round(c, self.float_pre).tolist()
            data_per_cam[color_name] = {
                'kp': kp,
                'c': c
            }
            # data_per_cam[color]['c'] = [lower[0], lower[1], lower[2], 1]
        if get_drawing_dict:
            for_drawing_dict['frame'] = frame
        return data_per_cam, for_drawing_dict

    @staticmethod
    def draw_on_frame(frame: np.array, data_per_cam: dict, drawing_dict_per_cam: dict) -> None:
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        # draw cx,cy on frame
        cv2.circle(frame, (int(frame_w / 2), int(frame_h / 2)), radius=15, color=utils.BGR_ORANGE, thickness=3)

        for color_name, corners_dict in drawing_dict_per_cam["rects"].items():
            cv2.rectangle(frame,
                          pt1=(corners_dict["min_x"], corners_dict["min_y"]),
                          pt2=(corners_dict["max_x"], corners_dict["max_y"]),
                          color=utils.BGR_YELLOW,
                          thickness=2)
        return

    @staticmethod
    def unpack_to_list_of_2d_points(data_all_ports: dict, frame_h: int):
        datum, colors_sets = [], []
        for cam_port, data_per_cam in data_all_ports.items():
            kps, colors = [], []
            for color_name, kp_and_c in data_per_cam.items():
                kp = np.copy(np.array(kp_and_c['kp'], dtype=float))  # 2d point
                if kp.shape[0] > 0:
                    kp[1] = frame_h - kp[1]
                    kps.append(kp)
                    c_rgba = kp_and_c['c']
                    colors.append(c_rgba)
            data = np.array(kps, dtype=float)
            datum.append(data)
            colors_sets.append(colors)
        return datum, colors_sets

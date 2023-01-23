# The detector based on the code here:
# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
import numpy as np
import cv2
import SharedCode.PY.utilsScript as utils
from Clients.CamClient.PY.Detectors.Colors import ColorsDetector
from Clients.CamClient.PY.Detectors.motion_detector import MotionDetector


class DroneDetector:
    def __init__(self,
                 motion_cfg: dict, mac: str, cam_port_to_cap: dict, backgrounds_folder: str,
                 # these are for motion detection
                 colors_cfg: dict,  # these are for color detection
                 float_pre: int):

        self.motion_detector = MotionDetector(
            mac=mac,
            cam_port_to_cap=cam_port_to_cap,
            backgrounds_folder=backgrounds_folder,
            min_area=motion_cfg["min_area"],
            gaus_dict=motion_cfg["gaus_dict"],
            thresh_dict=motion_cfg["thresh_dict"],
            dilate_dict=motion_cfg["dilate_dict"],
            padding=motion_cfg["padding"],
            float_pre=float_pre,
            show_bases=motion_cfg["show_bases"],
        )

        self.colors_detector = ColorsDetector(
            boundaries_bgr=colors_cfg['boundaries_bgr'],
            float_pre=float_pre
        )

        self.float_pre = float_pre
        return

    def __str__(self):
        string = 'DroneDetector:\n'
        string += '{}\n'.format('-' * 20)
        string += self.motion_detector.__str__()
        string += '{}\n'.format('-' * 20)
        string += self.colors_detector.__str__()
        string += '{}\n'.format('-' * 20)
        string += 'Members:\n'
        string += '\tfloat_pre={}\n'.format(self.float_pre)
        return string

    def detect_drone(self, frame: np.array, cam_port: int, get_drawing_dict: bool = False) -> (dict, dict):
        """ TODO
        """
        data_per_cam_colors = {}
        drawing_dict_per_cam = {"frame": frame}
        data_per_cam_motion, drawing_dict_per_cam_motion = self.motion_detector.detect_motion(
            frame, cam_port=cam_port, get_drawing_dict=get_drawing_dict)
        if get_drawing_dict:
            drawing_dict_per_cam["motion_data"] = data_per_cam_motion  # needed just for drawing
            drawing_dict_per_cam["motion_draw"] = drawing_dict_per_cam_motion

        if len(data_per_cam_motion) > 0:  # if found an object
            is_found_drone = False
            data_per_cam_colors, drawing_dict_per_cam_colors = None, None
            for motion_key in data_per_cam_motion.keys():  # iterate over every motion square until we find the drone
                min_x, min_y = data_per_cam_motion[motion_key]['kps']['tl']['kp']
                max_x, max_y = data_per_cam_motion[motion_key]['kps']['br']['kp']
                motion_box = frame[int(min_y):int(max_y), int(min_x):int(max_x)]

                data_per_cam_colors, drawing_dict_per_cam_colors = self.colors_detector.detect_colors(
                    motion_box, get_drawing_dict=get_drawing_dict)

                # need to fix x,y in respect to the real frame
                if len(data_per_cam_colors.keys()) == 2:
                    for color_name, kp_and_c in data_per_cam_colors.items():
                        kp_and_c['kp'][0] += min_x
                        kp_and_c['kp'][1] += min_y
                    data_per_cam_colors['center'] = {
                        'kp': [int((min_x + max_x) / 2), int((min_y + max_y) / 2)],
                        'c': [0.0, 1.0, 0.0, 1.0]
                    }
                    # we found the square with the colors, this is the drone, no need to look for other motion squares
                    is_found_drone = True
                    break
            if not is_found_drone:
                data_per_cam_colors, drawing_dict_per_cam_colors = {}, {}
            else:
                if get_drawing_dict:
                    # need to fix x,y in respect to the real frame
                    for color_name, corners_dict in drawing_dict_per_cam_colors["rects"].items():
                        # mb: motion box
                        min_x_mb, min_y_mb = corners_dict['min_x'], corners_dict['min_y']
                        max_x_mb, max_y_mb = corners_dict['max_x'], corners_dict['max_y']
                        corners_dict['min_x'] = int(min_x + min_x_mb)
                        corners_dict['max_x'] = int(min_x + max_x_mb)
                        corners_dict['min_y'] = int(min_y + min_y_mb)
                        corners_dict['max_y'] = int(min_y + max_y_mb)
                    drawing_dict_per_cam["colors_draw"] = drawing_dict_per_cam_colors
        print('data_per_cam_colors: ', data_per_cam_colors)
        return data_per_cam_colors, drawing_dict_per_cam

    @staticmethod
    def draw_on_frame(frame: np.array, data_per_cam: dict, drawing_dict_per_cam: dict) -> None:
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        # draw cx,cy on frame
        cv2.circle(frame, (int(frame_w / 2), int(frame_h / 2)), radius=15, color=utils.BGR_ORANGE, thickness=3)

        _ = MotionDetector.draw_on_frame(
            frame,
            drawing_dict_per_cam['motion_data'],
            drawing_dict_per_cam['motion_draw']
        )

        if len(data_per_cam) > 0:  # found drone
            ColorsDetector.draw_on_frame(
                frame,
                data_per_cam,
                drawing_dict_per_cam['colors_draw']
            )

        return

    @staticmethod
    def unpack_to_list_of_2d_points(data_all_ports: dict, frame_h: int):
        datum, colors_sets = ColorsDetector.unpack_to_list_of_2d_points(data_all_ports, frame_h)
        # datum, colors_sets = MotionDetector.unpack_to_list_of_2d_points(data_all_ports, frame_h)
        return datum, colors_sets

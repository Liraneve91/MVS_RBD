# The detector based on the code here:
# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
import numpy as np
import cv2
import SharedCode.PY.utilsScript as utils
from Clients.CamClient.PY.Detectors.Colors import ColorsDetector
import pickle
from Clients.CamClient.PY.Detectors.motion_detector import MotionDetector


class DroneDetector2:
    def __init__(self, cam_port_to_cap: dict, float_pre: int):
        self.float_pre = float_pre
        self.threshold = 250
        self.min_area = 0.1
        self.max_area = 100.0
        return

    def __str__(self):
        string = 'DroneDetector:\n'
        string += '{}\n'.format('-' * 20)
        string += 'Members:\n'
        string += '\tfloat_pre={}\n'.format(self.float_pre)
        return string

    def detect_drone(self, frame: np.array, cam_port: int, get_drawing_dict: bool = False) -> (dict, dict):
        data_per_cam_drone = {}
        drawing_dict_per_cam = {"frame": frame}
        for_drawing_dict = {"rects": {}} if get_drawing_dict else None

        # if cam_port == 0:
        #     # get updated location of objects in subsequent frames
        #     success, boxes = self.multiTracker_cam1.update(frame)
        # else:
        #     # get updated location of objects in subsequent frames
        #     success, boxes = self.multiTracker_cam2.update(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert BGR to Binary
        if cam_port==0:
            gray = gray[0:400 , :]
        if cam_port==2:
            gray = gray[:400 , :]
        ret, bin_frame = cv2.threshold(gray, self.threshold, 250, cv2.THRESH_BINARY)

        contours, h = cv2.findContours(bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        k_points2d = []
        if len(contours) > 0:
            for c in contours:
                moment = cv2.moments(c)
                if self.min_area < moment['m00'] < self.max_area:
                    point = np.array([[moment['m10'] / moment['m00'], moment['m01'] / moment['m00']]])  # 1x2 array
                    k_points2d.append(point)
            if len(k_points2d)>0:
                k_points2d = np.array(k_points2d, dtype='float64')
                k_points2d = k_points2d.reshape(k_points2d.shape[0], k_points2d.shape[2])  # change from x,1,2 to x,2
                # k_points2d = self.filter_centers_by_closest(k_points2d, cam_port)
        data_per_cam = {'k_points2d': np.round(k_points2d, self.float_pre).tolist()}



        p1 = (0,0)
        p2 = (0,0)
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0

        if len(k_points2d)>0:
            min_x = k_points2d[0][0]-10
            max_x = k_points2d[0][0]+10
            if cam_port==0:
                # min_y = k_points2d[0][1]-10+50
                # max_y = k_points2d[0][1]+10+50
                min_y = k_points2d[0][1]-10
                max_y = k_points2d[0][1]+10
            else:
                min_y = k_points2d[0][1] - 10
                max_y = k_points2d[0][1] + 10


        # draw tracked objects
        # for i, newbox in enumerate(boxes):
        #     p1 = (int(newbox[0]), int(newbox[1]))
        #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #     color =(255, 64, 64)
        #     cv2.rectangle(frame, p1, p2, color, 2, 1)
        # min_x = p1[0]
        # max_x = p2[0]
        # min_y = p1[1]
        # max_y = p2[1]

        # TODO: Fix cam 1 calibration
        # if cam_port==0:
        if True:
            if get_drawing_dict:
                for_drawing_dict["rects"]["center"] = {
                    "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y
                }

            if len(k_points2d)>0:
                data_per_cam_drone['center'] = {
                    'kp': [int(k_points2d[0][0]), int(k_points2d[0][1])],
                    'c': [0.0, 1.0, 0.0, 1.0]
                }

                if get_drawing_dict:
                    drawing_dict_per_cam["drone_tracker"] = for_drawing_dict  # needed just for drawing

        return data_per_cam_drone, drawing_dict_per_cam

    @staticmethod
    def draw_on_frame(frame: np.array, data_per_cam: dict, drawing_dict_per_cam: dict) -> None:
        if 'drone_tracker' in drawing_dict_per_cam:
            p1 = (int(drawing_dict_per_cam["drone_tracker"]["rects"]["center"]["min_x"]),
                  int(drawing_dict_per_cam["drone_tracker"]["rects"]["center"]["min_y"]))
            p2 = (int(drawing_dict_per_cam["drone_tracker"]["rects"]["center"]["max_x"]),
                  int(drawing_dict_per_cam["drone_tracker"]["rects"]["center"]["max_y"]))
            cv2.rectangle(img=frame, pt1=p1, pt2=p2, color=utils.BGR_GREEN, thickness=2)

        return

    @staticmethod
    def unpack_to_list_of_2d_points(data_all_ports: dict, frame_h: int):
        datum, colors_sets = ColorsDetector.unpack_to_list_of_2d_points(data_all_ports, frame_h)
        # datum, colors_sets = MotionDetector.unpack_to_list_of_2d_points(data_all_ports, frame_h)
        return datum, colors_sets

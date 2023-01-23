import cv2
import numpy as np
import matplotlib
import SharedCode.PY.utilsScript as utils

# chessboard black corners touching right hand:
# else ld -> ru etc.
CORNERS_DICT = {
    'ld': {'idx': 0, 'c': 'b'},  # left down
    'ld2': {'idx': 1, 'c': 'b'},  # left down on the right
    'lu': {'idx': 5, 'c': 'r'},  # left up
    'rd': {'idx': 48, 'c': 'g'},  # right up # for small chessboard
    'ru': {'idx': 53, 'c': 'y'},  # right down # for small chessboard
    # 'rd': {'idx': 24, 'c': 'g'},  # right up # for big chessboard
    # 'ru': {'idx': 29, 'c': 'y'},  # right down # for big chessboard
}


class ChessBoardDetector:
    def __init__(self, pattern_size_x: int, pattern_size_y: int, extract_colors: bool, float_pre: int = 2):
        self.pattern_size = (pattern_size_x, pattern_size_y)
        self.extract_colors = extract_colors
        self.float_pre = float_pre
        self.last_corners_found = None
        return

    def __str__(self):
        string = 'ChessBoardDetector:\n'
        string += '\tpattern_size={}\n'.format(self.pattern_size)
        string += '\textract_colors={}\n'.format(self.extract_colors)
        string += '\tfloat_pre={}\n'.format(self.float_pre)
        return string

    def detect_chessboard(self, frame: np.array, get_drawing_dict: bool = False) -> (dict, dict):
        """
        unpack:
        for corner_name, kp_and_c in data_per_cam.items():
            print(corner_name)
            kp = np.array(kp_and_c['kp'], dtype=float)  # 2d point
            if kp.shape[0] > 0:
                print('\tkp {}'.format(kp))
                if 'c' in kp_and_c:
                    c_rgba = kp_and_c['c']
                    print('\tc_rgba {}'.format(c_rgba))
        lu left up
        ld left down
        ru right up
        rd right down
        :return a dict with body part to kp.
        """
        data_per_cam = {}
        for_drawing_dict = None

        pattern_found, corners = cv2.findChessboardCorners(frame, self.pattern_size)
        if get_drawing_dict:
            for_drawing_dict = {
                'p_found': pattern_found,
                'frame': frame
            }

        if pattern_found:
            corners = corners.reshape(corners.shape[0], corners.shape[2])  # corners.shape[1] == 1
            corners = np.round(corners, self.float_pre)

            for corner_name, corner_meta in CORNERS_DICT.items():
                kp = corners[corner_meta['idx']]
                kp = np.round(kp.tolist(), self.float_pre).tolist()
                data_per_cam[corner_name] = {'kp': kp}
                if self.extract_colors:
                    c = matplotlib.colors.to_rgba(corner_meta['c'], alpha=None)
                    c = np.round(c, self.float_pre).tolist()
                    data_per_cam[corner_name]['c'] = c
            if get_drawing_dict:
                for_drawing_dict['p_size'] = self.pattern_size
                for_drawing_dict['corners'] = corners  # first save for drawing
        else:
            for corner_name, corner_meta in CORNERS_DICT.items():
                data_per_cam[corner_name] = {'kp': []}
                if self.extract_colors:
                    data_per_cam[corner_name]['c'] = []
        return data_per_cam, for_drawing_dict

    @staticmethod
    def draw_on_frame(frame: np.array, data_per_cam: dict, for_drawing_dict: dict) -> None:
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        # draw cx,cy on frame
        cv2.circle(frame, (int(frame_w / 2), int(frame_h / 2)), radius=15, color=utils.BGR_ORANGE, thickness=-1)

        # if len is 0: recoded data - no drawing dict - cant draw chess corners
        if 'p_found' in for_drawing_dict and for_drawing_dict['p_found']:
            corners = for_drawing_dict['corners']  # shape 54,2
            # DRAW colorful chessboard on frame
            cv2.drawChessboardCorners(frame, for_drawing_dict['p_size'], corners, True)

        for corner_name, kp_and_c in data_per_cam.items():
            kp = np.array(kp_and_c['kp'], dtype=float)  # 2d point
            if kp.shape[0] > 0:
                kp_cv = kp.T.reshape(1, 1, 2)  # reshape to 1,1,2
                contours_to_plot = np.array(kp_cv, dtype=np.int32)  # cast to int
                c_bgr = utils.BGR_GREEN  # default color
                if 'c' in kp_and_c:
                    c_rgba = np.array(kp_and_c['c'], dtype=float)
                    c_bgr = []
                    for j in range(len(c_rgba) - 1):  # -1 remove A
                        c_bgr.append(int(c_rgba[j] * 255))  # change from 0-1 -> 0-255
                    c_bgr.reverse()  # RGB TO BGR
                # DRAW keypoints(4) on frame
                cv2.drawContours(frame, contours_to_plot, -1, color=c_bgr, thickness=10)
        return

    @staticmethod
    def unpack_to_list_of_2d_points(data_all_ports: dict, frame_h: int):
        datum, colors_sets = [], []
        for cam_port, data_per_cam in data_all_ports.items():
            kps, colors = [], []
            for corner_name, kp_and_c in data_per_cam.items():
                kp = np.copy(np.array(kp_and_c['kp'], dtype=float))  # 2d point
                if kp.shape[0] > 0:
                    kp[1] = frame_h - kp[1]
                    kps.append(kp)
                    if 'c' in kp_and_c:
                        c_rgba = kp_and_c['c']
                        colors.append(c_rgba)
            data = np.array(kps, dtype=float)
            datum.append(data)
            colors_sets.append(colors)
        return datum, colors_sets

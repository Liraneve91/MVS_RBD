"""
based on
https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
"""
import os
import cv2
import numpy as np
import imutils
import SharedCode.PY.utilsScript as utils


class MotionDetector:
    def __init__(
            self,
            mac: str,
            cam_port_to_cap: dict,
            backgrounds_folder: str,
            min_area: int,
            gaus_dict: dict = None,
            thresh_dict: dict = None,
            dilate_dict: dict = None,
            padding: int = 20,
            float_pre: int = 2,
            show_bases: bool = False
    ):
        """
        :param mac: mac of the device
        :param cam_port_to_cap: all cams playing and the caps if they are active
        :param backgrounds_folder: empty background images folder.
                folder has 'mac'/'cam_port'/0.jpg for each port in ports
        :param min_area: min size of the contour - 1 will give you noise all the contours
        :param gaus_dict:
        :param thresh_dict:
        :param dilate_dict:
        :param padding:
        :param float_pre:
        :param show_bases:
        future: add flag send_area: send the contour area. for now always sending
                add flag send_img: send the sub image (for comparing). for now always sending
        """
        self.mac = mac
        self.cam_port_to_cap = cam_port_to_cap
        self.backgrounds_folder = backgrounds_folder
        self.min_area = min_area  # min size of the contour - 1 will give you noise

        # default value if received none
        self.gaus = gaus_dict if gaus_dict is not None else {'ksize': (21, 21), 'sigmaX': 1}
        self.gaus['ksize'] = tuple(self.gaus['ksize'])
        self.thresh = thresh_dict if thresh_dict is not None else {'thresh': 75}
        self.dilate = dilate_dict if dilate_dict is not None else {'kernel': None, 'iterations': 10}

        self.padding = padding
        self.float_pre = float_pre

        self.base_frames = {}

        for port, cap in self.cam_port_to_cap.items():
            if cap is not None:  # take bg image and save locally
                img_path = None
                success, img_bgr = cap.read()
                if not success:
                    print(utils.WARNING.format('failure capturing frame', 'self.cap'))
                # img_bgr = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
            else:  # use bg defined in the backgrounds_folder in the ds
                img_path = os.path.join(self.backgrounds_folder, str(self.mac).replace(':', ''), str(port), '0.jpg')
                assert os.path.exists(img_path), 'bg image not found for {}/{} at {}'.format(self.mac, port, img_path)
                img_bgr = cv2.imread(img_path)

            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # convert it to gray scale
            self.base_frames[port] = {
                'path': img_path,
                'img_bgr': img_bgr,
                'img_gray': img_gray
            }

        if show_bases:
            title = 'mac {}, '.format(self.mac)
            bases = []
            for port, base_f_dict in self.base_frames.items():
                bases.append(base_f_dict['img_bgr'])
                bases.append(base_f_dict['img_gray'])
                title += 'port {} (BGR, gray), '.format(port)
            utils.display_open_cv_images(imgs=bases, ms=0, title=title, x_y=(0, 0), resize=0.60,
                                         grid=(len(self.base_frames), 2))
        return

    def __str__(self):
        string = 'MotionDetector on mac={} on ports={}:\n'.format(self.mac, list(self.cam_port_to_cap.keys()))
        string += '\tbackgrounds_folder={}\n'.format(self.backgrounds_folder)
        string += '\tmin_area={}\n'.format(self.min_area)
        string += '\tgaus={}\n'.format(self.gaus)
        string += '\tthresh={}\n'.format(self.thresh)
        string += '\tdilate={}\n'.format(self.dilate)
        string += '\tpadding={}\n'.format(self.padding)
        string += '\tfloat_pre={}\n'.format(self.float_pre)
        for port, base_f_dict in self.base_frames.items():
            string += '\tcam={}\n'.format(port)
            string += '\t\tbase_frame_path={}\n'.format(base_f_dict['path'])
            string += '\t\t{}\n'.format(utils.var_to_string(base_f_dict['img_bgr'], 'img_bgr', with_data=False))
            string += '\t\t{}\n'.format(utils.var_to_string(base_f_dict['img_gray'], 'img_gray', with_data=False))
        return string

    def detect_motion(self, frame: np.array, cam_port: int, get_drawing_dict: bool = False) -> (dict, dict):
        """
        :param frame: img np array
        :param cam_port: port needed to compare to the corresponding background
        :param get_drawing_dict: if get_drawing_dict True, for_drawing_dict will hold data for drawing
        :return: data_per_cam:dict. each contour has an entry.
                    in each, 'kps': c_dict, 'image': sub_image, 'area': c_area
                    kps:dict. 'corner_name': {kp, c}
                    sub_image: np array - the contour img
                    area: contour area
                 for_drawing_dict frame, thresh img, and the rectangles(which can be calculated from data_per_cam)
        """
        for_drawing_dict = None if not get_drawing_dict else {'frame': frame}
        frame_h, frame_w = frame.shape[0], frame.shape[1]

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert it to gray scale
        frame_gray = cv2.GaussianBlur(src=frame_gray, ksize=self.gaus['ksize'], sigmaX=self.gaus['sigmaX'])  # blur it
        frames_delta_gray = cv2.absdiff(src1=self.base_frames[cam_port]['img_gray'], src2=frame_gray)
        delta_thresh_gray = cv2.threshold(
            src=frames_delta_gray, thresh=self.thresh['thresh'], maxval=255, type=cv2.THRESH_BINARY)[1]
        # dilate the threshold image to fill in holes, then find contours on threshold image
        delta_thresh_gray = cv2.dilate(src=delta_thresh_gray, kernel=self.dilate['kernel'],
                                       iterations=self.dilate['iterations'])
        contours = cv2.findContours(
            image=delta_thresh_gray,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(cnts=contours)
        # print('\t|contours|={}'.format(len(contours)))
        bounding_rects_cv, contours_dict, count = [], {}, 0
        for j, c in enumerate(contours):
            # print('\t\tarea = {}'.format(cv2.contourArea(c)))
            c_area = cv2.contourArea(contour=c)
            if c_area >= self.min_area:  # if the contour is too small, ignore it
                # print('\t\tcontour {} passed. area = {}'.format(j, c_area))
                # compute the bounding box for the contour, draw it on the frame, and update the text
                (x, y, w, h) = cv2.boundingRect(array=c)
                x = max(x - self.padding, 0)
                y = max(y - self.padding, 0)
                w = min(w + 2 * self.padding, frame_w)
                h = min(h + 2 * self.padding, frame_h)
                bounding_rects_cv.append((x, y, w, h))  # x,y:= top left corner of the rectangle
                # cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=utils.BGR_GREEN, thickness=2)

                sub_image = frame[int(y):int(y + h), int(x):int(x + w)].tolist()
                # print(utils.var_to_string(sub_image, 'si'))
                # utils.display_open_cv_image(sub_image, 1)
                c_dict = {  # prepare contour data: 4 2d points, area dict, sub image
                    # t,b: top and bottom. l,r: left right
                    'tl': {'kp': np.array([x, y], dtype=float).tolist()},
                    'bl': {'kp': np.array([x, y + h], dtype=float).tolist()},
                    'tr': {'kp': np.array([x + w, y], dtype=float).tolist()},
                    'br': {'kp': np.array([x + w, y + h], dtype=float).tolist()},
                    'center': {'kp': np.array([x + int(w/2), y + int(h/2)], dtype=float).tolist()},
                }
                contours_dict['c{}'.format(count)] = {
                    'kps': c_dict,
                    'image': sub_image,
                    'area': c_area,  # if needed to take top area contour
                }
                count += 1
        data_per_cam = contours_dict
        if get_drawing_dict:
            for_drawing_dict['rects_cv'] = bounding_rects_cv
            for_drawing_dict['delta_thresh_gray'] = delta_thresh_gray
        return data_per_cam, for_drawing_dict

    @staticmethod
    def draw_on_frame(frame: np.array, data_per_cam: dict, for_drawing_dict: dict) -> np.array:
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        # draw cx,cy on frame
        cv2.circle(frame, (int(frame_w / 2), int(frame_h / 2)), radius=15, color=utils.BGR_ORANGE, thickness=3)

        for c_ind, c_data in data_per_cam.items():
            pt1 = tuple(np.array(c_data['kps']['tl']['kp'], dtype=np.int32))
            pt2 = tuple(np.array(c_data['kps']['br']['kp'], dtype=np.int32))
            cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=utils.BGR_GREEN, thickness=2)

        dtg = None if for_drawing_dict is None else for_drawing_dict['delta_thresh_gray']
        return dtg

    @staticmethod
    def unpack_to_list_of_2d_points(data_all_ports: dict, frame_h: int):
        datum, colors_sets = [], []
        for cam_port, data_per_cam in data_all_ports.items():
            kps, colors = [], []
            for c_ind, c_data in data_per_cam.items():
                for kp_name, kp_dict in c_data['kps'].items():
                    if kp_name in ['center']:
                        kp = np.copy(np.array(kp_dict['kp'], dtype=float))  # 2d point
                        kp[1] = frame_h - kp[1]
                        kps.append(kp)
            data = np.array(kps, dtype=float)
            datum.append(data)
            colors_sets.append(colors)
        return datum, colors_sets


def get_next_frame(ds_path: str, cam_port: int, t: int) -> np.array:
    full_img_path = '{}/{}/{}.jpg'.format(ds_path, cam_port, t)
    assert os.path.exists(full_img_path), 'file {} not found'.format(full_img_path)
    frame = cv2.imread(full_img_path)
    print('\tread img from: {}'.format(full_img_path))
    return frame


def main():
    ds_path = '../../../../SharedCode/Calib/2021_02_08_11_41_26/ds_drone_colors/CamClientInput/DCA632BE8E8D'
    max_rounds = 30
    scatters = []
    plot_x_y_lim = [0, 640, 0, 480]
    frame_h = None

    cam_port_to_cap = {
        0: None,
        1: None
    }

    md = MotionDetector(
        mac='DCA632BE8E8D',
        cam_port_to_cap={0: None, 1: None},
        backgrounds_folder='../../../../SharedCode/Calib/2021_02_08_11_41_26/ds_drone_colors/empty_lab_backgrounds',
        min_area=1000,
        gaus_dict={
            "ksize": [21, 21],
            "sigmaX": 1
        },
        thresh_dict={"thresh": 75},
        dilate_dict={
            "kernel": None,
            "iterations": 10
        },
        padding=20,
        float_pre=2,
        show_bases=True,
    )
    print(md)

    for t in range(max_rounds):
        print('round {}'.format(t))
        # frame = get_next_frame(ds_path, i)  # BGR frame
        # data_per_cam, for_drawing_dict = md.detect_movement(frame, get_drawing_dict=True)
        # dtg = md.draw_on_frame(frame, data_per_cam, for_drawing_dict)
        # utils.display
        # _open_cv_images(imgs=[frame, dtg, frame, dtg], ms=0, title='x', x_y=(0, 0), resize=60, grid=(2, 2))

        data_dict = {}  # will hold work output for each camera port
        drawing_dict = {}  # will hold drawing data if show open cv true for each cam

        for cam_port, cap in cam_port_to_cap.items():
            frame = get_next_frame(ds_path, cam_port, t)  # BGR frame
            frame_h = frame.shape[0]
            data_per_cam, drawing_dict_per_cam = md.detect_motion(frame, cam_port, get_drawing_dict=True)
            drawing_dict[cam_port] = drawing_dict_per_cam
            data_dict[cam_port] = data_per_cam

        final_rgb_frames = []

        # individual draws (cam has all the data to draw)
        for cam_idx, data_per_cam in data_dict.items():
            drawing_dict_per_cam = drawing_dict[cam_idx]
            frame = drawing_dict_per_cam['frame']
            delta_thresh_gray = MotionDetector.draw_on_frame(frame, data_per_cam, drawing_dict_per_cam)
            final_rgb_frames.append(frame)
            final_rgb_frames.append(delta_thresh_gray)

        title = 'mac {} cams {}'.format(md.mac, list(cam_port_to_cap.keys()))
        x_y = (0, 70) if t == 0 else None
        utils.display_open_cv_images(imgs=final_rgb_frames, ms=1, title=title, x_y=x_y, resize=0.60, grid=(2, 2))

        datum, colors_sets = md.unpack_to_list_of_2d_points(data_dict, frame_h)
        block = (t == max_rounds - 1)  # t is the last iteration
        title = 'mac {} cams {} iteration {}'.format(md.mac, list(cam_port_to_cap.keys()), t)
        if len(scatters) == 0:  # build plot first time
            titles = []
            for cam_port in data_dict.keys():
                titles.append('cam {}'.format(cam_port))
            scatters = utils.plot_horizontal_figures(datum, colors_sets, titles, main_title=title,
                                                     x_y_lim=plot_x_y_lim, block=block,
                                                     plot_location='top_right')
        else:
            utils.update_scatters(scatters, datum, colors_sets, new_title=title, block=block)

    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    from wizzi_utils import all as wu

    # wu.first_func()
    wu.main_wrapper(
        main_function=main,
        cuda_off=False,
        torch_v=False,
        tf_v=False,
        cv2_v=True,
        with_profiler=False
    )

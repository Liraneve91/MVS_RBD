import json
import numpy as np
import atexit
import os
import threading
import math
import time
import subprocess
import SharedCode.PY.utilsScript as utils
from Clients.Client.PY.ClientClass import Client
from Clients.CamClient.PY.Detectors.ChessBoard import ChessBoardDetector
from Clients.CamClient.PY.Detectors.Colors import ColorsDetector
from Clients.CamClient.PY.Detectors.motion_detector import MotionDetector
from Clients.CamClient.PY.Detectors.Drone import DroneDetector
from Clients.CamClient.PY.Detectors.Drone2 import DroneDetector2
from Clients.CamClient.PY.Detectors.IrBlobs import IrBlobsDetector
from SharedCode.PY import recordedDataLoader

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
import cv2  # noqa: E402

PROJECT_HOME = '../../../'  # relative to CamMain.py
PROJECT_ABS_PATH = os.path.abspath(PROJECT_HOME)
print('PROJECT_ABS_PATH: {}'.format(PROJECT_ABS_PATH))
os.chdir(PROJECT_ABS_PATH)

CAM_CONSTANTS_PATH = 'Clients/CamClient/CamConstants.json'

CFG_FILES = [utils.SHARED_CONSTANTS_PATH, CAM_CONSTANTS_PATH]
for cfg_file in CFG_FILES:
    assert os.path.exists(cfg_file), 'cant find {}'.format(cfg_file)

assert os.path.exists(utils.SERVER_ADDRESS_PATH), 'cant find {}'.format(utils.SERVER_ADDRESS_PATH)

DEVICE_TYPE = 'RP'  # 'RP' , 'jetson nano'
MAX_IMAGES_FOR_QUEUE = 5
IS_USE_DISTORTION = False


class CamHandler(Client):
    def __init__(self):
        np.random.seed(42)
        print('CamHandler::CamHandler()')
        cfg = utils.load_jsons(CFG_FILES, ack=False)
        server_json = utils.load_json(utils.SERVER_ADDRESS_PATH, ack=False)
        if cfg['misc_cam']['use_fake_mac']:
            self.mac = cfg['misc_cam']['fake_mac']
        else:
            self.mac = utils.get_mac_address()

        self.cam_port_to_cap, self.cam_port_to_images, self.record_output = {}, {}, False  # for dtor if con fail
        super().__init__(
            cfg=cfg,
            local_mode=cfg['misc_cam']['local_mode_cfg']['local_mode'],
            send_output_data_from_ds=cfg['misc_cam']['send_output_data_from_ds'],
            server_json=server_json,
            name='CamClient',
            extra_dict={'mac': self.mac}
        )

        cv_image_cfg = cfg["misc_cam"]['show_images']['cv']
        self.cv_image = cv_image_cfg['show']
        self.cv_image_ms = cv_image_cfg['wait_ms']
        self.cv_title = None
        self.cv_x_y = tuple(cv_image_cfg['x_y'])
        self.cv_resize = cv_image_cfg['resize']
        self.cv_grid = None

        plt_image_cfg = cfg["misc_cam"]['show_images']['plt']
        self.plt_image = plt_image_cfg['show']
        self.plt_x_y_lim = list(plt_image_cfg['x_y_limits'].values())
        self.plt_location = plt_image_cfg['plot_window_location']
        self.plt_resize = plt_image_cfg['resize']
        self.plt_title = None
        self.plt_scatters = []  # used for showing the plot of the output data

        self.calib_root = None
        self.chess_board_detector = None
        self.ir_blobs_detector = None
        self.colors_detector = None
        self.motion_detector = None
        self.drone_detector = None

        self.sleep_between_round = self.cfg['misc_cam']['sleep_between_rounds']
        self.cameras_params = utils.CameraParams.read_calib_data(self.cfg['calib_data'], self.mac, True)
        self.skip_read_frames = self.cfg["misc_cam"]["skip_read_frames"]
        self.images_base_path, self.images_size = None, None  # images variables
        self.record_output, self.recorded_output = self.cfg["misc_cam"]["record_output"], {}  # record variables
        self.loaded_output_data, self.out_ds_size = None, None  # out data variables
        atexit.register(self.clean_up)
        return

    def clean_up(self):
        print('CamHandler::clean_up()')
        if self.record_output:
            ds_full_path = recordedDataLoader.get_cam_client_output_data_path(
                self.ds_path, self.detection_mode, self.mac)
            recorded_file_name = '{}cams_{}iters.pkl'.format(len(self.cam_port_to_cap), len(self.recorded_output))
            file_path = os.path.join(ds_full_path, recorded_file_name)
            utils.save_pkl(file_path, self.recorded_output, ack=True)
        for cam_port, cap in self.cam_port_to_cap.items():
            if cap is not None:
                cap.release()
        return

    def __del__(self):
        print('CamHandler::~CamHandler()')
        return

    def initialize(self):
        print('Extra data post server ctor')
        self.do_work(iteration_t='shake msg 4/5', expected_msg='post_ctor_server')
        print('Cam Initialization:')
        max_ports_on_device = self.cfg['misc_cam']['max_cams_on_device']
        if self.run_mode in ['calibration', 'recording_ds']:
            self.calib_root = self.cfg['calib_params']['calib_root']
            utils.create_dir_if_not_exists(self.calib_root)
            self.calib_root = os.path.join(self.calib_root, utils.get_current_time_stamp())
            utils.create_dir_if_not_exists(self.calib_root)
            self.calib_root = os.path.join(self.calib_root, self.mac.replace(':', ''))
            utils.create_dir_if_not_exists(self.calib_root)

            ports = [0, 2]
            # for i in range(max_ports_on_device):
            for i in ports:
                utils.create_dir_if_not_exists(os.path.join(self.calib_root, str(i)))
                if self.skip_read_frames:
                    self.cam_port_to_cap[i] = None
                else:
                    cap = None
                    if DEVICE_TYPE == 'jetson nano':
                        cap_text = utils.gstreamer_pipeline(cam_port=i, flip_method=0)
                        cap = cv2.VideoCapture(cap_text, cv2.CAP_GSTREAMER)
                    elif DEVICE_TYPE == 'RP':
                        cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        self.cam_port_to_cap[i] = cap
                        print('\t\tcap is open on port {}: {}'.format(i, cap))
                    else:
                        print('\t\tno camera detected on port {}'.format(i))
        elif self.run_mode == 'mapping':
            if self.detection_mode == "drone2":
                subprocess.check_call("v4l2-ctl -d /dev/video0 -c white_balance_temperature_auto=1", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_auto=1", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video0 -c gain=0", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=50", shell=True)
                # subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_auto=0", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video0 -c focus_auto=0", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video0 -c contrast=100", shell=True)

                subprocess.check_call("v4l2-ctl -d /dev/video2 -c white_balance_temperature_auto=1", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video2 -c exposure_auto=1", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video2 -c gain=0", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video2 -c exposure_absolute=40", shell=True)
                # subprocess.check_call("v4l2-ctl -d /dev/video2 -c exposure_auto=0", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video2 -c focus_auto=0", shell=True)
                subprocess.check_call("v4l2-ctl -d /dev/video2 -c contrast=100", shell=True)
            err_msg = 'this mac={}. calib macs {}'.format(self.mac, self.cfg['calib_data'].keys())
            assert self.mac in self.cfg['calib_data'], err_msg

            # try to open all port from the entry
            rp_calib_data = self.cfg['calib_data'][self.mac]
            print('\tmy calib data: {}'.format(rp_calib_data))  # just for extra debugging
            ports = [0, 2]
            # for cam_port_str, cam_calib_data in rp_calib_data.items():
            for cam_port_str in ports:
                cam_port_int = int(cam_port_str)
                if self.skip_read_frames:
                    self.cam_port_to_cap[cam_port_int] = None
                else:
                    cap = None
                    if DEVICE_TYPE == 'jetson nano':
                        cap = cv2.VideoCapture(
                            utils.gstreamer_pipeline(cam_port=cam_port_int, flip_method=0),
                            cv2.CAP_GSTREAMER)
                    elif DEVICE_TYPE == 'RP':
                        cap = cv2.VideoCapture(cam_port_int)
                    if cap.isOpened():
                        self.cam_port_to_cap[cam_port_int] = cap
                        print('\t\tself.cap is open on port {}: {}'.format(cam_port_int, cap))
                    else:
                        print(utils.WARNING.format('failed to open on port {}'.format(cam_port_int), 'camClient'))
            # valid only on detection mode (not calibrate)
            if self.send_output_data_from_ds:
                self.loaded_output_data, self.out_ds_size = recordedDataLoader.load_cam_client_output_data(
                    self.ds_path, self.detection_mode, self.mac)
            self.initialize_detectors()

        if self.skip_read_frames:  # must be after ctor - need first cam port
            self.images_base_path, self.images_size = self.create_images_base_path()

        if self.cv_image:  # done once
            self.cv_title = 'mac {} cams {}'.format(self.mac, list(self.cam_port_to_cap.keys()))
            utils.display_open_cv_images(imgs=[[[1], [1], [1]]], ms=1, title=self.cv_title,
                                         x_y=self.cv_x_y, resize=0, grid=(1, 1))
            self.cv_x_y = None

        if self.plt_image:
            self.plt_title = 'mac {} cams {}'.format(self.mac, list(self.cam_port_to_cap.keys()))
            self.plt_title += ' iteration {}'
            titles = []
            datum, colors_sets = [], []
            for cam_port in self.cam_port_to_cap.keys():
                titles.append('cam {}'.format(cam_port))
                datum.append([])
                colors_sets.append([])

            self.plt_scatters = utils.plot_horizontal_figures(datum, colors_sets, titles,
                                                              main_title=self.plt_title.format(-1),
                                                              x_y_lim=self.plt_x_y_lim, block=False,
                                                              plot_location=self.plt_location,
                                                              resize=self.plt_resize)

        print('\tm_run_mode={}'.format(self.run_mode))
        print('\tm_cam_port_to_cap={}'.format(self.cam_port_to_cap))
        print('\tm_local_mode={}'.format(self.local_mode))
        print('\tm_skip_read_frames={}'.format(self.skip_read_frames))
        if self.skip_read_frames:
            print('\t\timages size={}'.format(self.images_size))
        print('\tsend_output_data_from_ds={}'.format(self.send_output_data_from_ds))
        if self.send_output_data_from_ds:
            print('\t\tout ds size={}'.format(self.out_ds_size))
        print('\tm_detection_mode={}'.format(self.detection_mode))
        return

    def initialize_detectors(self):
        if self.detection_mode == 'chessboard':
            chess_cfg = self.cfg['detector_params']['chessboard']
            self.chess_board_detector = ChessBoardDetector(
                pattern_size_x=chess_cfg['pattern_size_x'],
                pattern_size_y=chess_cfg['pattern_size_y'],
                extract_colors=chess_cfg['with_colors'],
                float_pre=self.float_pre
            )
            print(self.chess_board_detector)

        elif self.detection_mode == 'blobs':
            blobs_cfg = self.cfg['detector_params']['blobs']
            self.ir_blobs_detector = IrBlobsDetector(
                threshold=blobs_cfg['threshold_value'],
                min_area=blobs_cfg['min_area'],
                max_area=blobs_cfg['max_area'],
                # todo pass from server k after connection is made
                K=1,
                float_pre=self.float_pre
            )
            print(self.ir_blobs_detector)

        elif self.detection_mode == 'colors':
            colors_cfg = self.cfg["detector_params"]["colors"]
            self.colors_detector = ColorsDetector(
                boundaries_bgr=colors_cfg['boundaries_bgr'],
                float_pre=self.float_pre
            )
            print(self.colors_detector)

        elif self.detection_mode == 'motion':
            motion_cfg = self.cfg["detector_params"]["motion"]

            self.motion_detector = MotionDetector(
                mac=self.mac,
                cam_port_to_cap=self.cam_port_to_cap,
                backgrounds_folder=os.path.join(self.ds_path, motion_cfg["backgrounds_folder"]),
                min_area=motion_cfg["min_area"],
                gaus_dict=motion_cfg["gaus_dict"],
                thresh_dict=motion_cfg["thresh_dict"],
                dilate_dict=motion_cfg["dilate_dict"],
                padding=motion_cfg["padding"],
                float_pre=self.float_pre,
                show_bases=motion_cfg["show_bases"],
            )
            print(self.motion_detector)

        elif self.detection_mode == 'drone':
            # drone_cfg = self.cfg["detector_params"]["drone"]  # if needed - insert to CFG
            motion_cfg = self.cfg["detector_params"]["motion"]
            colors_cfg = self.cfg["detector_params"]["colors"]

            self.drone_detector = DroneDetector(
                motion_cfg=motion_cfg,
                mac=self.mac,
                cam_port_to_cap=self.cam_port_to_cap,
                backgrounds_folder=os.path.join(self.ds_path, motion_cfg["backgrounds_folder"]),
                colors_cfg=colors_cfg,
                float_pre=self.float_pre,
            )
            print(self.drone_detector)

        elif self.detection_mode == 'drone2':
            self.drone_detector2 = DroneDetector2(
                cam_port_to_cap=self.cam_port_to_cap, float_pre=self.float_pre,
            )
            print(self.drone_detector)
        return

    def start_capture_images_for_cam_port(self, cam_port):
        while True:
            success, frame = self.cam_port_to_cap[cam_port].read()
            if not success:
                print(utils.WARNING.format('failure capturing frame', 'self.cap'))
                exit(-1)
            self.cam_port_to_images[cam_port].append(frame)
            if len(self.cam_port_to_images[cam_port]) >= MAX_IMAGES_FOR_QUEUE:
                del self.cam_port_to_images[cam_port][0]

    def get_fake_msg_from_server(self, iteration_t: int, expected_msg: str) -> (str, int):
        j_in = {'name': 'Server', 'msg': expected_msg, 't': iteration_t}
        fake_data_in = utils.json_to_string(j_in)
        return fake_data_in, len(fake_data_in)

    def do_client_specific_work(self, j_in: json) -> json:
        t = j_in['t']
        j_out = {}
        if j_in['msg'] == 'work':
            data_dict = self.handle_work(j_in)
            j_out = {'name': self.name, 'msg': 'output', 'data': data_dict}
        elif j_in['msg'] == 'post_ctor_server':
            # expecting  j_in['extra'] to be empty
            extra_out = {}  # nothing to send
            j_out = {'name': self.name, 'msg': 'post_ctor_client', 't': 'shake msg 5/5', 'extra': extra_out}
        elif j_in['msg'] == 'calib_take_img':
            responses_dict = self.take_image(t)
            j_out = {'name': self.name, 'msg': 'calib_image_taken', 't': t, 'data': responses_dict}
        elif j_in['msg'] == 'calib_send_img':
            imgs_dict = self.load_image(t)
            j_out = {'name': self.name, 'msg': 'calib_output_img', 't': t, 'data': imgs_dict}
        return j_out

    def get_next_frame(self, cam_port: int, cap, t: int, cam_port_to_frame: dict) -> np.array:
        if self.skip_read_frames:
            full_img_path = '{}/{}/{}.jpg'.format(self.images_base_path, cam_port, t % self.images_size)
            assert os.path.exists(full_img_path), 'file {} not found'.format(full_img_path)
            frame = cv2.imread(full_img_path)
            if self.debug:
                print('\t\tHandling cam {}. read img from: {}'.format(cam_port, full_img_path))
        else:
            success, frame = cap.read()
            if not success:
                print(utils.WARNING.format('failure capturing frame', 'self.cap'))
                while(success!=True):
                    cap.release()
                    cap = cv2.VideoCapture(cam_port)
                    success, frame = cap.read()
        if IS_USE_DISTORTION and self.detection_mode != 'calibration':
            intrinsic = self.cameras_params[cam_port].intrinsic
            distortion = self.cameras_params[cam_port].distortion
            (w, h) = frame.shape[0:2]
            frame = np.float32(frame)
            distortion = np.float32(distortion)
            scaled_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (w, h), 1, (w, h))
            frame = cv2.undistort(frame, intrinsic, distortion, None, scaled_cam_matrix)
        cam_port_to_frame[cam_port] = np.uint8(frame)
        return

    def handle_work(self, j_in: json) -> dict:
        t = j_in['t']
        data_dict = {}  # will hold work output for each camera port
        drawing_dict = {}  # will hold drawing data if show open cv true for each cam
        frame_h = None

        threads_dict = {}
        cam_port_to_frame = {}

        for cam_port, cap in self.cam_port_to_cap.items():
            threads_dict[cam_port] = threading.Thread(target=self.get_next_frame,
                                                      args=(cam_port, cap, t, cam_port_to_frame))
            threads_dict[cam_port].start()
            print('\tstarted thread of cam: {}'.format(cam_port))
            # self.get_images_detection_details_single_thread(image_to_detection_details, cam_path, obj_p)
        for cam_port, cap in self.cam_port_to_cap.items():
            threads_dict[cam_port].join()
            print('\tfinished thread of cam: {}'.format(cam_port))

        for cam_port, cap in self.cam_port_to_cap.items():
            frame = cam_port_to_frame[cam_port]  # self.get_next_frame(cam_port, cap, t)
            frame_h = frame.shape[0]
            data_per_cam, drawing_dict_per_cam = {}, {}

            if self.send_output_data_from_ds:
                data_per_cam = self.loaded_output_data[t % self.out_ds_size][cam_port]
                drawing_dict[cam_port] = {'frame': frame}
            else:
                if self.detection_mode == "chessboard":
                    data_per_cam, drawing_dict_per_cam = self.chess_board_detector.detect_chessboard(
                        frame, get_drawing_dict=self.cv_image)
                elif self.detection_mode == "blobs":
                    data_per_cam, drawing_dict_per_cam = self.ir_blobs_detector.detect_ir_blobs(
                        frame, cam_port, get_drawing_dict=self.cv_image)
                elif self.detection_mode == "colors":
                    data_per_cam, drawing_dict_per_cam = self.colors_detector.detect_colors(
                        frame, get_drawing_dict=self.cv_image)
                elif self.detection_mode == "motion":
                    data_per_cam, drawing_dict_per_cam = self.motion_detector.detect_motion(
                        frame, cam_port=cam_port, get_drawing_dict=self.cv_image)
                elif self.detection_mode == "drone":
                    data_per_cam, drawing_dict_per_cam = self.drone_detector.detect_drone(
                        frame, cam_port=cam_port, get_drawing_dict=self.cv_image)
                elif self.detection_mode == "drone2":
                    data_per_cam, drawing_dict_per_cam = self.drone_detector2.detect_drone(
                        frame, cam_port=cam_port, get_drawing_dict=self.cv_image)
                # if self.debug:
                #     print(utils.var_to_string(data_per_cam, '\t\tdata_per_cam', with_data=True))

                if self.cv_image:
                    # extra data for drawing. e.g. (frame, for chessboard save corners ...). can be None
                    drawing_dict[cam_port] = drawing_dict_per_cam
            data_dict[cam_port] = data_per_cam

        if self.cv_image:
            final_frames = self.draw_on_frames(data_dict, drawing_dict, t=t)
            utils.display_open_cv_images(imgs=final_frames, ms=self.cv_image_ms, title=self.cv_title,
                                         x_y=self.cv_x_y, resize=self.cv_resize, grid=self.cv_grid)

        if self.plt_image:
            datum, colors_sets, block = self.unpack_data(data_dict, t, frame_h)
            utils.update_scatters(self.plt_scatters, datum, colors_sets, new_title=self.plt_title.format(t),
                                  block=block)

        if self.record_output:
            self.recorded_output[t] = data_dict

        if self.sleep_between_round > 0:
            time.sleep(self.sleep_between_round)
        return data_dict

    def take_image(self, t: int) -> dict:
        responses_dict, imgs_for_im_show = {}, []
        t=t+0
        threads_dict = {}
        cam_port_to_frame = {}
        for cam_port, cap in self.cam_port_to_cap.items():
            threads_dict[cam_port] = threading.Thread(target=self.get_next_frame,
                                                      args=(cam_port, cap, t, cam_port_to_frame))
            threads_dict[cam_port].start()
            print('started thread of cam: ', cam_port)
            # self.get_images_detection_details_single_thread(image_to_detection_details, cam_path, obj_p)
        for cam_port, cap in self.cam_port_to_cap.items():
            threads_dict[cam_port].join()
            print('finished thread of cam: ', cam_port)

        for cam_port, cap in self.cam_port_to_cap.items():
            if cap is not None and cap.isOpened and DEVICE_TYPE == "RP":  # if image don't refresh - this worked
                cap.release()
                cap.open(cam_port)
            img = cam_port_to_frame[cam_port]  # self.get_next_frame(cam_port, cap, t)
            print(utils.var_to_string(img, '\t\t\timg {} port {}'.format(t, cam_port)))
            utils.save_img(path=os.path.join(self.calib_root, str(cam_port), '{}.jpg'.format(t)),
                           img=img, ack=True, tabs=3)
            imgs_for_im_show.append(img)
            responses_dict[cam_port] = 'success'

        if self.cfg["calib_params"]["show_image"]:
            title = 'CamClient {} iter {} ({} images)'.format(self.mac, t, len(imgs_for_im_show))
            utils.display_open_cv_images(imgs_for_im_show, ms=self.cfg["calib_params"]["show_image_time"], title=title,
                                         resize=0.5)
        return responses_dict

    def load_image(self, t: int) -> dict:
        imgs_dict = {}
        for cam_port, cap in self.cam_port_to_cap.items():
            img = utils.load_img(path=os.path.join(self.calib_root, str(cam_port), '{}.jpg'.format(t)),
                                 ack=True, tabs=2)
            print(utils.var_to_string(img, '\t\t\timg {} port {}'.format(t, cam_port)))
            imgs_dict[cam_port] = img.tolist()
        return imgs_dict

    def create_images_base_path(self):
        ds_path = self.ds_path
        assert os.path.exists(ds_path), 'ds not found - {}'.format(ds_path)
        mac_clean = self.mac.replace(':', '')
        images_base_path = os.path.join(ds_path, 'CamClientInput', mac_clean)
        assert os.path.exists(images_base_path), 'skip_read_frames True but no images dir found on {}'.format(
            images_base_path)
        first_cam_port = list(self.cam_port_to_cap.keys())[0]
        path, dirs, files = next(os.walk('{}/{}'.format(images_base_path, first_cam_port)))
        images_size = len(files)  # take size from first cam folder
        assert images_size > 0, 'no images in folder: {}'.format(images_base_path)
        return images_base_path, images_size

    def draw_on_frames(self, data_dict: dict, drawing_dict: dict, t: int) -> (list, list):
        final_frames = []  # RGB OR GRAY

        # individual draws (cam has all the data to draw)
        for cam_idx, data_per_cam in data_dict.items():
            drawing_dict_per_cam = drawing_dict[cam_idx]  # could be None
            frame = drawing_dict_per_cam['frame']

            if self.detection_mode == "chessboard":
                ChessBoardDetector.draw_on_frame(frame, data_per_cam, drawing_dict_per_cam)
                final_frames.append(frame)
            elif self.detection_mode == "blobs":
                thresh1_rgb = IrBlobsDetector.draw_on_frame(frame, data_per_cam, drawing_dict_per_cam)
                final_frames.append(frame)
                if thresh1_rgb is not None:
                    final_frames.append(thresh1_rgb)
            elif self.detection_mode == "colors":
                ColorsDetector.draw_on_frame(frame, data_per_cam, drawing_dict_per_cam)
                final_frames.append(frame)
            elif self.detection_mode == "motion":
                delta_thresh_gray = MotionDetector.draw_on_frame(frame, data_per_cam, drawing_dict_per_cam)
                final_frames.append(frame)
                if delta_thresh_gray is not None:
                    final_frames.append(delta_thresh_gray)
            elif self.detection_mode == "drone":
                DroneDetector.draw_on_frame(frame, data_per_cam, drawing_dict_per_cam)
                final_frames.append(frame)
            elif self.detection_mode == "drone2":
                DroneDetector2.draw_on_frame(frame, data_per_cam, drawing_dict_per_cam)
                final_frames.append(frame)

        # done once. can be done after first time we know hom many frames each cam produced (post iter 0)
        if self.cv_grid is None and self.cv_image:
            frames_n = len(final_frames)
            if frames_n < 4:
                self.cv_grid = (1, frames_n)
            else:  # each cam has a row
                cams_n = len(self.cam_port_to_cap.keys())
                self.cv_grid = (cams_n, math.ceil(frames_n / cams_n))

        # # non individual draws (need 2 cams to draw)
        # if self.detection_mode == 'orb':  # TODO restore later
        #     images_matched = OrbsDetector.draw_matches_between_2_frames(
        #         data_dict, drawing_dict, best_orbs_num=self.cfg['detector_params']['orb']['best_orbs_num'])
        #     if images_matched is not None:
        #         extra_frames.append(images_matched)
        return final_frames

    def unpack_data(self, data_dict, t, frame_h):
        datum, colors_sets = [], []
        if self.detection_mode in ["chessboard"]:
            datum, colors_sets = ChessBoardDetector.unpack_to_list_of_2d_points(data_dict, frame_h)
        elif self.detection_mode in ["blobs"]:
            datum, colors_sets = IrBlobsDetector.unpack_to_list_of_2d_points(data_dict, frame_h)
        elif self.detection_mode in ["motion"]:
            datum, colors_sets = MotionDetector.unpack_to_list_of_2d_points(data_dict, frame_h)
        elif self.detection_mode in ["colors"]:
            datum, colors_sets = ColorsDetector.unpack_to_list_of_2d_points(data_dict, frame_h)
        elif self.detection_mode in ["drone"]:
            datum, colors_sets = DroneDetector.unpack_to_list_of_2d_points(data_dict, frame_h)

        block = (
                self.local_mode and  # only in local mode there is a 'last iteration'
                (t == self.cfg['misc_cam']['local_mode_cfg']['max_iters'] - 1) and  # t is the last iteration
                self.cfg['misc_cam']['local_mode_cfg']['block_last_iter']  # user ask to block
        )
        return datum, colors_sets, block


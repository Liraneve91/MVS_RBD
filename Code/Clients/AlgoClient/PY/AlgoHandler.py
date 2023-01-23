import os
import json
import itertools
import random

import cv2
import atexit
import numpy as np
import SharedCode.PY.utilsScript as utils
from Clients.AlgoClient.PY import KMeansScript
from Clients.Client.PY.ClientClass import Client
import SharedCode.PY.recordedDataLoader as recordedDataLoader
from collections import OrderedDict

PROJECT_HOME = '../../../'  # relative to AlgoMain.py
PROJECT_ABS_PATH = os.path.abspath(PROJECT_HOME)
print('PROJECT_ABS_PATH: {}'.format(PROJECT_ABS_PATH))
os.chdir(PROJECT_ABS_PATH)
MAC_STR_SIZE = 12
MAC_STR_JUMP = 2
ALGO_CONSTANTS_PATH = 'Clients/AlgoClient/AlgoConstants.json'

CFG_FILES = [utils.SHARED_CONSTANTS_PATH, ALGO_CONSTANTS_PATH]
for cfg_file in CFG_FILES:
    assert os.path.exists(cfg_file), 'cant find {}'.format(cfg_file)

assert os.path.exists(utils.SERVER_ADDRESS_PATH), 'cant find {}'.format(utils.SERVER_ADDRESS_PATH)


class AlgoHandler(Client):
    def __init__(self):
        np.random.seed(42)
        print('AlgoHandler::AlgoHandler()')
        cfg = utils.load_jsons(CFG_FILES, ack=False)
        server_json = utils.load_json(utils.SERVER_ADDRESS_PATH, ack=False)
        self.record_output = False  # for dtor if con fail
        super().__init__(
            cfg=cfg,
            local_mode=cfg['misc_algo']['local_mode_cfg']['local_mode'],
            send_output_data_from_ds=cfg['misc_algo']['send_output_data_from_ds'],
            server_json=server_json,
            name='AlgoClient',
        )
        assert 'calib_data' in cfg, 'No calib data found on cfg'  # Algo cant run without a calib file
        self.rps_playing = None  # mac numbers, initialized post connection
        self.cam_count = 0
        self.cams_info_dict = OrderedDict()
        self.all_total_cost_results = []
        self.kmeans_algorithm = None
        self.send_lines = self.cfg["misc_algo"]["with_lines"]
        self.loaded_input_data, self.input_ds_size = None, None  # in data variables
        self.record_output, self.recorded_output = self.cfg["misc_algo"]["record_output"], {}  # record variables
        self.loaded_output_data, self.out_ds_size = None, None  # out data variables
        atexit.register(self.clean_up)
        return

    def clean_up(self):
        print('CamHandler::clean_up()')
        if self.record_output:
            ds_full_path = recordedDataLoader.get_algo_client_output_data_path(self.ds_path, self.detection_mode)
            recorded_file_name = '{}devices_{}cams_{}iters.pkl'.format(len(self.rps_playing), self.cam_count,
                                                                       len(self.recorded_output))
            file_path = os.path.join(ds_full_path, recorded_file_name)
            utils.save_pkl(file_path, self.recorded_output, ack=True)
        return

    def __del__(self):
        print('AlgoHandler::~AlgoHandler()')
        return

    def initialize(self):
        print('Extra data post server ctor')
        self.do_work(iteration_t='shake msg 4/5', expected_msg='post_ctor_server')
        print('Initializing KMeans...')  # create CameraInfo for each cam on each device
        cameras_params = utils.CameraParams.read_calib_data(self.cfg['calib_data'], self.rps_playing, self.debug)
        self.calib_data = self.cfg['calib_data']
        self.graph_edges = self.cfg['graph']['g_edges']
        self.cam_count = len(cameras_params)
        cams_positions = []
        for camera_params in cameras_params:
            cam_info_i = KMeansScript.CameraInfo(
                mac=camera_params.mac,
                cam_port=camera_params.port,
                position=camera_params.p,
                rotation=camera_params.R,
                intrinsic=camera_params.intrinsic
            )
            print('{}'.format(cam_info_i))
            cams_positions.append(cam_info_i.position)
            self.cams_info_dict[camera_params.get_id(delimiter=self.mac_p_del)] = cam_info_i
        cams_positions = np.array(cams_positions, dtype=float)

        kmeans_params_cfg = self.cfg["kmeans"][self.detection_mode]
        run_type = kmeans_params_cfg['run_type']
        self.kmeans_algorithm = KMeansScript.KMeans(
            k=kmeans_params_cfg['K'],
            run_type=run_type,
            t=kmeans_params_cfg['t'],
            use_cam_pos_filter=kmeans_params_cfg['use_cam_pos_filter'],
            use_in_range_filter=kmeans_params_cfg['use_in_range_filter'],
            use_unique_points_filter=kmeans_params_cfg['use_unique_points_filter'],
            use_noise_filter=kmeans_params_cfg['use_noise_filter'],
            cams_positions=cams_positions,
            in_range_values=kmeans_params_cfg['in_range_values'],
            noise_filter_size=kmeans_params_cfg['noise_filter_size'],
            debug=self.debug if run_type != '1_mean_acc' else False  # loop over kmeans - no need for prints
        )

        if self.local_mode:
            self.loaded_input_data, self.input_ds_size = recordedDataLoader.load_algo_client_input_data(
                self.ds_path, self.detection_mode, self.rps_playing)

        if self.send_output_data_from_ds:
            self.loaded_output_data, self.out_ds_size = recordedDataLoader.load_algo_client_output_data(
                self.ds_path, self.detection_mode)

        print(self.kmeans_algorithm)
        print('rps_playing: {}'.format(self.rps_playing))
        print('m_run_mode={}'.format(self.run_mode))
        print('m_local_mode={}'.format(self.local_mode))
        if self.local_mode:
            print('\tin ds size={}'.format(self.input_ds_size))
        print('send_output_data_from_ds={}'.format(self.send_output_data_from_ds))
        if self.send_output_data_from_ds:
            print('\tout ds size={}'.format(self.out_ds_size))
        print('m_detection_mode={}'.format(self.detection_mode))
        return

    def get_fake_msg_from_server(self, iteration_t: int, expected_msg: int) -> (str, int):
        j_in = {}
        if expected_msg == 'post_ctor_server':
            all_rp_macs = []  # takes first 'local_mode_rps_playing' rps with all their cams from the calib file
            for i, cam_mac in enumerate(self.cfg["calib_data"].keys()):
                all_rp_macs.append(cam_mac)
            assert len(all_rp_macs) > 0, "no rps found on calib file"
            max_rps = min(self.cfg['misc_algo']['local_mode_cfg']["rps_playing"], len(all_rp_macs))
            rps_playing = all_rp_macs[:max_rps]  # macs of rps that will connect
            extra = {'rps_playing': rps_playing}
            j_in = {'name': 'Server', 'msg': 'post_ctor_server', 'extra': extra, 't': iteration_t}
        elif expected_msg == 'work':
            data_dict = self.loaded_input_data[iteration_t % self.input_ds_size]  # t starts from 1
            j_in = {'name': 'Server', 'msg': 'work', 'data': data_dict, 't': iteration_t}

        fake_data_in = utils.json_to_string(j_in)
        return fake_data_in, len(fake_data_in)

    def do_client_specific_work(self, j_in: json) -> json:
        j_out = {}
        if j_in['msg'] == 'work':
            data_dict = self.handle_work(j_in)
            j_out = {'name': self.name, 'msg': 'output', 'data': data_dict}
        elif j_in['msg'] == 'post_ctor_server':
            self.rps_playing = j_in['extra']['rps_playing']
            extra_out = {}  # nothing to send
            j_out = {'name': self.name, 'msg': 'post_ctor_client', 't': 'shake msg 5/5', 'extra': extra_out}
        return j_out

    def handle_work(self, j_in: json) -> dict:
        t = j_in['t']

        if self.send_output_data_from_ds:  # recorded data mode
            data_dict_out = self.loaded_output_data[t % self.out_ds_size]
        else:  # real data mode
            data_dict_in = j_in['data']  # got input - insert to the CameraInfo and run kmeans
            data_dict_out = {'d3points': []}
            total_cost = float("inf")
            if self.detection_mode == "chessboard":
                data_dict_out, total_cost = self.handle_chessboard_data(data_dict_in)

            elif self.detection_mode == "motion":
                data_dict_out, total_cost = self.handle_motion_data(data_dict_in)

            elif self.detection_mode == "colors":
                data_dict_out, total_cost = self.handle_colors_data(data_dict_in)

            elif self.detection_mode == "drone":
                data_dict_out, total_cost = self.handle_drone_data(data_dict_in)

            elif self.detection_mode == "drone2":
                data_dict_out, total_cost = self.handle_drone_data(data_dict_in)

            self.all_total_cost_results.append(total_cost)
            print('\t\ttotal cost={:.2f}, avg={:.2f}'.format(total_cost, np.mean(self.all_total_cost_results)))

            data_dict_out['d3points'] = np.round(data_dict_out['d3points'], self.float_pre).tolist()
            P = np.array(data_dict_out['d3points'])
            print('all distances:',
                  [round(np.linalg.norm(pair[0] - pair[1], ord=2), 3) for pair in itertools.product(P, repeat=2)])
            if 'colors' in data_dict_out:
                data_dict_out['colors'] = np.round(data_dict_out['colors'], self.float_pre).tolist()

        if self.debug:
            d3points = np.array(data_dict_out['d3points'], dtype='float64')
            print(utils.var_to_string(d3points, '\t\td3points', with_data=True))
            # for d3p in d3points:
            #     print('\t\t\td3p {}'.format(d3p.tolist()))
            # if 'lines' in data_dict_out:
            #     print('\t\tlines:')
            #     for line_dict in data_dict_out['lines']:
            #         print('\t\t\tp={}, v={}'.format(line_dict['p'], line_dict['v']))
            if 'colors' in data_dict_out:
                colors = np.array(data_dict_out['colors'], dtype='float64')
                print(utils.var_to_string(colors, '\t\tcolors', with_data=True))

        if self.record_output:
            self.recorded_output[t] = data_dict_out

        return data_dict_out

    def match_parts_and_run_kmeans(self, dict_per_object: dict, parts_names: list) -> (dict, float):
        """
        this func will work if dict_per_object has only one object from many MACs
        :param dict_per_object:
            chess shape:
            {mac1: ..., mac2: ..., macFinal}
            dict_per_object['mac1'] = {0: ..., 1: ..., lastPortOnMac: ...}
            dict_per_object['mac1'][0] = {part1: ..., part2: ..., lastPart: ...}
            dict_per_object['mac1'][0][part1] = { 'kp':np.array of type float, 'c': optional list }
            skeletons shape:
            {mac1: ..., mac2: ..., macFinal}
            dict_per_object['mac1'] = {0: ..., 1: ..., lastPortOnMac: ...}
            dict_per_object['mac1'][0] = {0(sk1): ..., 1(sk2): ..., 2(sk_last_detected): ...}
            dict_per_object['mac1'][0][0(sk1)] = {part1: ..., part2: ..., lastPart: ...}
        :param parts_names: all possible parts name to be matched
        :return:
        """
        # # SECOND iterate on each name, set points, call kmeans (1_mean_acc)
        total_cost, d3points, colors = 0.0, [], []
        lines_list = []
        for part_name_in_progress in parts_names:
            c_rgba = None
            valid_cam_info = 0  # we need at least 2 cam info that detected the chessboard
            for mac, cam_to_data in dict_per_object.items():
                for cam_port, data_per_cam in cam_to_data.items():
                    for part_name, kp_and_c in data_per_cam.items():
                        if part_name == part_name_in_progress:
                            kp = np.array(kp_and_c['kp'], dtype=float)  # 2d point
                            if kp.shape[0] > 0:
                                valid_cam_info += 1
                                # same color for all same corner_name - just take first 1 seen if there is color
                                if c_rgba is None and 'c' in kp_and_c:
                                    c_rgba = kp_and_c['c']
                            cam_key = '{}{}{}'.format(mac, self.mac_p_del, cam_port)
                            self.cams_info_dict[cam_key].set_new_points(np.expand_dims(kp, axis=0))
            # print('there are {} valid cam info'.format(valid_cam_info))
            # print('color of this point_name {}'.format(c_rgba))
            if valid_cam_info > 1:
                lines = self.kmeans_algorithm.generate_lines_from_cams_info(cams_info_list=self.cams_info_dict.values())
                if len(lines) < 2:
                    continue
                d3point, cost = self.kmeans_algorithm.run_kmeans(lines)
                total_cost += cost
                d3points.append(d3point.tolist())
                if c_rgba is not None:
                    colors.append(c_rgba)
                if self.send_lines:
                    for line in lines:
                        line_dict = {
                            'p': np.round(line.p.flatten(), self.float_pre).tolist(),
                            'p_orig': np.round(line.p_orig.flatten(), self.float_pre).tolist(),
                            'v': np.round(line.v.flatten(), self.float_pre).tolist()
                        }
                        lines_list.append(line_dict)
        data_dict_out = {'d3points': d3points}
        if self.send_lines:
            data_dict_out['lines'] = lines_list
        # data_dict_out['d3points'] = np.round(np.array(d3points, dtype='float64'), self.float_pre).tolist()
        if len(colors) > 0:
            data_dict_out['colors'] = colors
            # data_dict_out['colors'] = np.round(np.array(colors, dtype='float64'), self.float_pre).tolist()
        return data_dict_out, total_cost

    def handle_chessboard_data(self, data_dict_in: dict) -> (dict, float):
        corner_names = []  # FIRST collect all points names
        for mac, cam_to_data in data_dict_in.items():
            for cam_port, data_per_cam in cam_to_data.items():
                for part_name, kp_and_c in data_per_cam.items():
                    if part_name not in corner_names:
                        corner_names.append(part_name)
        data_dict_out, total_cost = self.match_parts_and_run_kmeans(data_dict_in, parts_names=corner_names)
        return data_dict_out, total_cost

    def handle_colors_data(self, data_dict_in: dict) -> (dict, float):
        colors_names = []  # FIRST collect all points names
        for mac, cam_to_data in data_dict_in.items():
            for cam_port, data_per_cam in cam_to_data.items():
                for color_name, kp_and_c in data_per_cam.items():
                    if color_name not in colors_names:
                        colors_names.append(color_name)
        data_dict_out, total_cost = self.match_parts_and_run_kmeans(data_dict_in, parts_names=colors_names)
        return data_dict_out, total_cost

    def handle_drone_data(self, data_dict_in: dict) -> (dict, float):
        colors_names = []  # FIRST collect all points names
        for mac, cam_to_data in data_dict_in.items():
            for cam_port, data_per_cam in cam_to_data.items():
                for color_name, kp_and_c in data_per_cam.items():
                    if color_name in ['center']:
                        if color_name not in colors_names:
                            colors_names.append(color_name)
        data_dict_out, total_cost = self.match_parts_and_run_kmeans(data_dict_in, parts_names=colors_names)
        return data_dict_out, total_cost

    def handle_motion_data(self, data_dict_in: dict) -> (dict, float):
        # flatten all contours keys to a single unique ones and collect all kp_names
        c_k_list = []  # list of contours
        kp_names = []  # FIRST collect all points names

        for mac, cam_to_data in data_dict_in.items():
            for cam_port, data_per_cam in cam_to_data.items():
                for c_ind, c_data in data_per_cam.items():
                    key = '{}_{}_{}'.format(mac, cam_port, c_ind)
                    c_k_list.append(key)
                    for kp_name, kp_dict in c_data['kps'].items():
                        if kp_name not in kp_names:
                            kp_names.append(kp_name)
        n_choose_2 = list(itertools.combinations(c_k_list, 2))
        if self.debug:
            print('\t\tkp_names: {}'.format(kp_names))
            print('\t\tc_k_list: {}'.format(c_k_list))
            print('\t\tn_choose_2: {}'.format(n_choose_2))

        # calculate all matches
        pair_to_acc = {}
        for c1_k, c2_k in n_choose_2:
            mac1, port1, c_ind1 = c1_k.split('_')
            mac2, port2, c_ind2 = c2_k.split('_')
            if mac1 == mac2 and port1 == port2:  # cant be same skeleton from same mac and camera
                acc = 0.0
            else:
                acc = self.compare_two_images(
                    data_dict_in[mac1][port1][c_ind1]['image'],
                    data_dict_in[mac2][port2][c_ind2]['image'],
                    show_match=self.cfg['misc_algo_detectors']['motion']['show_match']
                )
            pair_to_acc['{}_p_{}'.format(c1_k, c2_k)] = acc

        if self.debug:
            print('\t\tpair_to_acc:')
            for pair, acc in pair_to_acc.items():
                c1_k, c2_k = pair.split('_p_')
                print('\t\t\tc1({}), c2({}), acc={}'.format(c1_k, c2_k, acc))

        # split to buckets
        # in theory - all contours are different - not 1 match
        buckets = [[] for _ in range(len(c_k_list))]  # create empty len(c_k_list) buckets
        match_threshold = self.cfg['misc_algo_detectors']['motion']['match_threshold']

        # TODO NOTICE - very naive greedy approach - replace to smarter bucketing
        for current_c_k in c_k_list:
            for bucket_i in buckets:
                belong_to_bucket_i = True
                for c_k_in_bucket in bucket_i:
                    if '{}_p_{}'.format(current_c_k, c_k_in_bucket) in pair_to_acc:
                        acc = pair_to_acc['{}_p_{}'.format(current_c_k, c_k_in_bucket)]
                    else:
                        acc = pair_to_acc['{}_p_{}'.format(c_k_in_bucket, current_c_k)]
                    if acc < match_threshold:
                        belong_to_bucket_i = False

                if belong_to_bucket_i:  # if empty bucket or all pairs in bucket acc above match_threshold
                    bucket_i.append(current_c_k)
                    break

        if self.debug:
            print('\t\tc_keys_buckets: {}'.format(buckets))

        # Now make a dict from each bucket and get 3points for this bucket(matched contours)
        c_dicts_list = []
        for bucket in buckets:
            if len(bucket) >= 2:  # ignore buckets that have less than 2 entries
                c_dict = {}
                for c_k in bucket:
                    mac, port, c_ind = c_k.split('_')
                    if mac not in c_dict:
                        c_dict[mac] = {}
                    c_dict[mac][port] = data_dict_in[mac][port][c_ind]['kps']
                c_dicts_list.append(c_dict)

        # construct 3d points out of the matched skeletons
        data_dict_out = {'d3points': []}
        if self.send_lines:
            data_dict_out['lines'] = []
        total_cost = 0.0
        for c_dict in c_dicts_list:
            # print(sk_dict)
            data_dict_out_1sk, total_cost_1sk = self.match_parts_and_run_kmeans(c_dict, parts_names=kp_names)
            total_cost += total_cost_1sk
            data_dict_out['d3points'].extend(data_dict_out_1sk['d3points'])
            if self.send_lines:
                data_dict_out['lines'].extend(data_dict_out_1sk['lines'])
            if 'colors' in data_dict_out_1sk:
                if 'colors' not in data_dict_out:  # first init []
                    data_dict_out['colors'] = []
                data_dict_out['colors'].extend(data_dict_out_1sk['colors'])
        return data_dict_out, total_cost

    def get_noisy_cloud_from_point(self, mean, color):
        cloud_size = random.choice(range(5, 20, 1))
        d3points_final, colors_final = [], []
        for i in range(int(cloud_size)):
            point = []
            for j in range(3):
                try:
                    val = random.choice(range(int(mean[j] - 2), int(mean[j] + 2), 1))
                except Exception as e:
                    print('mean: ', mean)
                    print('mean[j]: ', mean[j])
                    print('int(mean[j] - 2): ', int(mean[j] - 2))
                    print('int(mean[j] + 2): ', int(mean[j] + 2))
                    exit(-12)
                point.append(val)
            d3points_final.append(point)
            colors_final.append(color)
        return d3points_final, colors_final

    @staticmethod
    def compare_two_images(img1: list, img2: list, show_match: bool = False) -> float:
        img1, img2 = np.array(img1).astype(np.uint8), np.array(img2).astype(np.uint8)
        min_y, min_x = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (min_x, min_y), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, (min_x, min_y), interpolation=cv2.INTER_CUBIC)
        # TODO: consider other interpolation on images resize for a better matching
        error = cv2.matchTemplate(img1, img2, cv2.TM_SQDIFF_NORMED)[0][0]
        acc = 1 - error  # 0<= error <=1, error==1-acc
        if show_match:
            utils.display_open_cv_images([img1, img2], ms=1, title='acc={:.3f}'.format(acc), destroy_windows=False)
        return acc

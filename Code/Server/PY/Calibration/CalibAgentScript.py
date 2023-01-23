"""
intrinsic
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
extrinsic
https://www.programcreek.com/python/example/89371/cv2.stereoCalibrate
"""
import os
import sys
import SharedCode.PY.utilsScript as utils
import numpy as np

import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import networkx as nx
import itertools
import random
from networkx.algorithms.distance_measures import center
import os.path
from os import path
import pickle
import threading

MAC_STR_SIZE = 12
MAC_STR_JUMP = 2
# MAC_PORT_MIN_CHARS==4 - will display F9/1 instead of DC:A6:32:BE:B8:F9/1
# MAC_PORT_MIN_CHARS the minimum amount of chars needed to be UNIQUE
# set to >20 to get full mac/port
MAC_PORT_MIN_CHARS = 4
RELATIVE_EXTRINSIC_SINGLE_PAIR_MAX_IMAGES = 400
MIN_IMAGES_FOR_EDGE = 1
MAX_SECS_BETWEEN_IMAGES = 0.05  # 0.01 original
VIEWS_THREADS_NUMBER = 3
INTRINSIC_MAX_IMGS = 600
CALIB_ERROR_THRESHOLD = 0.2
IS_IMAGES_TO_DETECTION_FROM_PICKLE = True
IS_CALIB_DATA_FROM_PICKLE = True
IS_VIEWS_FROM_PICKLE = True
IS_ALL_RELATIVE_PAIRS_FROM_PICKLE = False
INT, EXT = 'intrinsic', 'extrinsic'
CALIB_MODE = EXT  # INT # EXT


class CalibAgent:
    def __init__(
            self,
            calib_folder: str,
            float_pre: int,
            show_corners: int,
            show_plot: int,
            show_views_graph: int,
            square_size: int,
            squares_num_width: int,
            squares_num_height: int
    ):
        """
        :param calib_folder: path to calib folder(calib file will be save here).
            also, it contains a folder named 'CalibImages':
            CalibImages structure:
                mac1
                    0 # port 0
                       img00.jpg
                       img01.jpg
                       ...
                    1 ...
                mac2 ...
            e.g. 'DCA632B81B90/0'. Notice this folder must exist in CalibImages
        :param float_pre:
        :param show_corners: show chessboard detection in extrinsic past
        :param show_plot: show cameras positions and direction plot
        :param square_size: size of square in the chessboard
        :param squares_num_width: number of squares vertically
        :param squares_num_height: number of squares horizontally
        """
        self.calib_folder = calib_folder
        assert os.path.exists(self.calib_folder), 'cant find {}'.format(self.calib_folder)
        self.calib_images_folder = os.path.join(self.calib_folder, 'CalibImages/' + CALIB_MODE)
        assert os.path.exists(self.calib_images_folder), 'cant find {}'.format(self.calib_images_folder)
        self.ground_cam = None
        self.ground_cam_suffix = None
        self.square_size = square_size
        self.squares_num_height = squares_num_height
        self.squares_num_width = squares_num_width
        self.squares_num_w_and_h = (self.squares_num_width, self.squares_num_height)
        self.float_pre = float_pre
        self.show_corners = show_corners
        self.show_plot = show_plot
        self.show_views_graph = show_views_graph
        self.axes, self.label_x, self.label_y, self.colors = None, 0.05, 0.95, None
        return

    def __str__(self):
        string = 'CalibAgent:\n'
        string += '\tcalib_folder={}\n'.format(self.calib_folder)
        string += '\tcalib_images_folder={}\n'.format(self.calib_images_folder)
        string += '\tground_cam={}\n'.format(self.ground_cam)
        string += '\tsquare_size={}\n'.format(self.square_size)
        string += '\tsquares_num_height={}\n'.format(self.squares_num_height)
        string += '\tsquares_num_width={}\n'.format(self.squares_num_width)
        string += '\tsquares_num_w_and_h={}\n'.format(self.squares_num_w_and_h)
        string += '\tfloat_pre={}\n'.format(self.float_pre)
        string += '\tshow_corners={}\n'.format(self.show_corners)
        string += '\tshow_plot={}\n'.format(self.show_plot)
        return string

    def build_plot(self, location: str = None):
        fig = plt.figure()
        if location is not None:
            utils.move_plot(fig, where=location)
        axes = Axes3D(fig)
        axes.autoscale(enable=True, axis='both', tight=True)
        axes.set_facecolor("black")
        cube_bottom, cube_top = -50, 50
        axes.set_xlim3d([cube_bottom, cube_top])
        axes.set_ylim3d([cube_bottom, cube_top])
        axes.set_zlim3d([cube_bottom, cube_top])
        label_color = 'r'
        axes.set_xlabel("X", color=label_color)
        axes.set_ylabel("Y", color=label_color)
        axes.set_zlabel("Z", color=label_color)
        axes.grid(False)
        background_and_alpha = (0.5, 0.2, 0.5, 0.4)  # RGB and alpha
        axes.w_xaxis.set_pane_color(background_and_alpha)
        axes.w_yaxis.set_pane_color(background_and_alpha)
        axes.w_zaxis.set_pane_color(background_and_alpha)
        utils.add_cube(axes, edge_len=40, add_corners_labels=False)
        axes.view_init(azim=-83, elev=-150)
        self.axes = axes
        return

    def update_plot(self, cam_ind: int, camera_params, c: str):
        p = camera_params.p.reshape(utils.DIM_3, 1)
        xs, ys, zs = p[0], p[1], p[2]
        # self.axes.plot3D(xs, ys, zs, color=c, marker="2", markersize=10, label='cams', linestyle="")
        self.axes.scatter(xs, ys, zs, color=c, s=20, label='cams')
        lbl = '{} {}'.format(camera_params.get_id()[-MAC_PORT_MIN_CHARS:],
                             np.round(p, self.float_pre).flatten().tolist())
        x, y, z = camera_params.p
        self.axes.text(x, y, z, camera_params.get_id()[-MAC_PORT_MIN_CHARS:], color=c, zdir=(1, 1, 1))
        self.axes.text2D(self.label_x, self.label_y - cam_ind * 0.03, lbl, transform=self.axes.transAxes, color=c)
        return

    def get_images_detection_from_cam_path(self, image_to_detection_details: dict, cam_path: str):
        mac_str = cam_path.split('/')[0]
        cam_port_str = cam_path.split('/')[1]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        glob_mac_1 = os.path.join(self.calib_images_folder, cam_path, '*.jpg')
        images = glob.glob(glob_mac_1)
        random.shuffle(images)
        count_valid, count_total, gray = 0, 0, None
        images_size = len(images)
        for image in images:
            count_total += 1
            print('\tget_images_detection cam_path: {}. finished until now: {}%'.format(cam_path, round(
                (float(count_total) / float(images_size) * 100.0), 3)))
            img = cv2.imread(image)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.squares_num_w_and_h, None)
            # If found, add object points, image points (after refining them)
            if ret:
                image_to_detection_details[image] = {}
                image_to_detection_details[image]['ret'] = ret
                image_to_detection_details[image]['image_shape'] = gray.shape[::-1]
                image_to_detection_details[image]['mac'] = mac_str
                image_to_detection_details[image]['cam_port'] = cam_port_str
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                image_to_detection_details[image]['corners'] = corners
                image_to_detection_details[image]['corners2'] = corners
                count_valid += 1

                if self.show_corners:  # Draw and display the corners
                    image_to_detection_details[image]['image'] = img
                    img = cv2.drawChessboardCorners(img, self.squares_num_w_and_h, corners, ret)

            if self.show_corners:  # Draw and display the corners
                cv2.imshow('img', img)
                cv2.waitKey()
        if self.show_corners:  # Draw and display the corners
            cv2.destroyAllWindows()
        print('\tcam {}: found Chessboard {}/{}'.format(cam_path[-MAC_PORT_MIN_CHARS:], count_valid, len(images)))
        assert count_valid > 0, 'cam {}: chessboard was not detected at all '.format(cam_path)
        return

    def get_images_detection_details(self, is_images_detection_from_pickle: bool, all_cams: list):
        image_to_detection_details = None
        image_to_detection_details_pickle_file_path = self.calib_folder + '/image_to_detection_details_' + CALIB_MODE + \
                                                      '.pickle'
        if is_images_detection_from_pickle:
            if path.exists(image_to_detection_details_pickle_file_path):
                print('\tloading image_to_detection_details from file..')
                image_to_detection_details = pickle.load(open(image_to_detection_details_pickle_file_path, "rb"))
            else:
                print('\timage_to_detection_details file not exists')
                exit(-1)
        else:
            print('\tcomputing image_to_detection_details from scratch..')
            image_to_detection_details = {}
            threads_dict = {}
            for cam_path in all_cams:
                threads_dict[cam_path] = threading.Thread(target=self.get_images_detection_from_cam_path,
                                                          args=(image_to_detection_details, cam_path))
                threads_dict[cam_path].start()
                print('started thread of cam: ', cam_path)
            for cam_path in all_cams:
                threads_dict[cam_path].join()
                print('finished thread of cam: ', cam_path)

            pickle.dump(image_to_detection_details, open(image_to_detection_details_pickle_file_path, "wb"))
            print('\timage_to_detection_details was saved to file: ', image_to_detection_details_pickle_file_path)
        print('\timage_to_detection_details dict is ready')
        return image_to_detection_details

    def get_calib_data_from_cam_path(self, cam_path: str, calib_data_json: dict, image_to_detection_details: dict,
                                     obj_p: np.ndarray, max_imgs: int):
        mac_str = cam_path.split('/')[0]
        cam_port_str = cam_path.split('/')[1]
        image_shape = None
        obj_points, img_points, count_valid = [], [], 0
        count_total, images_size = 0, len(image_to_detection_details)
        keys = list(image_to_detection_details.keys())
        random.shuffle(keys)
        for key in keys:
            value = image_to_detection_details[key]
            count_total += 1
            print('\tget_calib_data cam_path: {}. finished until now: {}%'.format(cam_path, round(
                (float(count_total) / float(images_size) * 100.0), 3)))
            current_mac = value['mac']
            current_cam_port = value['cam_port']
            if mac_str == current_mac and cam_port_str == current_cam_port:
                if value['ret']:
                    obj_points.append(obj_p)
                    img_points.append(value['corners2'])
                    if image_shape is None:
                        image_shape = value['image_shape']
                    count_valid += 1
                    if count_valid >= max_imgs:
                        break
        if len(obj_points) > 0:
            _, int_mat, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_shape, None, None)

            cam_intrinsic_params = {
                'intrinsic_mat': np.round(int_mat, self.float_pre).tolist(),
                'distortion': np.round(dist, self.float_pre).tolist(),
                'r': np.round(rvecs, self.float_pre).tolist(),
                't': np.round(tvecs, self.float_pre).tolist(),
            }
            calib_data_json['calib_data'][mac_str][cam_port_str].update(cam_intrinsic_params)
            print('\tcomputed intrinsic of camera: ', cam_path, ', images number: ', len(obj_points))
        return

    def calc_intrinsic_params(self, calib_data_json: dict, image_to_detection_details: dict, all_cams: list,
                              max_imgs: int, is_calib_data_from_pickle: bool = False):
        print('-' * 80)
        print('Calculating intrinsic parameters(max {} images for each):'.format(max_imgs))
        obj_p = np.zeros((self.squares_num_width * self.squares_num_height, 3), np.float32)
        obj_p[:, :2] = np.mgrid[0:self.squares_num_width, 0:self.squares_num_height].T.reshape(-1, 2) * self.square_size

        calib_data_json_pickle_file_path = self.calib_folder + '/calib_data_json.pickle'
        if not is_calib_data_from_pickle:
            print('\tcomputing calib_data_json from scratch..')
            threads_dict = {}
            for cam_path in all_cams:
                threads_dict[cam_path] = threading.Thread(target=self.get_calib_data_from_cam_path,
                                                          args=(
                                                              cam_path, calib_data_json, image_to_detection_details,
                                                              obj_p,
                                                              max_imgs))
                threads_dict[cam_path].start()
                print('\tstarted thread of cam: ', cam_path)
                # self.get_images_detection_details_single_thread(image_to_detection_details, cam_path, obj_p)
            for cam_path in all_cams:
                threads_dict[cam_path].join()
                print('\tfinished thread of cam: ', cam_path)

            pickle.dump(calib_data_json, open(calib_data_json_pickle_file_path, "wb"))
            print('\tcalib_data_json was saved to file: ', calib_data_json_pickle_file_path)
        else:
            # temp_get_intrinsic(all_cams, calib_data_json, self.float_pre)
            if path.exists(calib_data_json_pickle_file_path):
                print('\tloading calib_data_json from file..')
                calib_data_json = pickle.load(open(calib_data_json_pickle_file_path, "rb"))
            else:
                print('\tcalib_data_json_pickle_file_path.pickle does not exist')
                exit(-1)
        calib_data_json_final = {'calib_data': {}}
        for cam_path in all_cams:
            mac_str = cam_path.split('/')[0]
            cam_port_str = cam_path.split('/')[1]
            if mac_str not in calib_data_json_final['calib_data'].keys():
                calib_data_json_final['calib_data'][mac_str] = {}
            if cam_port_str not in calib_data_json_final['calib_data'][mac_str].keys():
                calib_data_json_final['calib_data'][mac_str][cam_port_str] = {}
            if 'distortion' not in calib_data_json_final['calib_data'][mac_str][cam_port_str].keys():
                calib_data_json_final['calib_data'][mac_str][cam_port_str]['distortion'] = \
                    calib_data_json['calib_data'][mac_str][cam_port_str]['distortion']
            if 'intrinsic_mat' not in calib_data_json_final['calib_data'][mac_str][cam_port_str].keys():
                calib_data_json_final['calib_data'][mac_str][cam_port_str]['intrinsic_mat'] = \
                    calib_data_json['calib_data'][mac_str][cam_port_str]['intrinsic_mat']
            print('\t\tcam_path: ', cam_path, '. distortion: ',
                  calib_data_json_final['calib_data'][mac_str][cam_port_str]['distortion'],
                  '. intrinsic_mat: ', calib_data_json_final['calib_data'][mac_str][cam_port_str]['intrinsic_mat'], '.')
        print('-' * 80)
        return calib_data_json_final

    def get_viwes_from_images_keys(self, views: dict, image_to_detection_details: dict,
                                   selected_keys: list, start_key_index: int):
        count_total, images_size = 0, len(selected_keys)
        for key in selected_keys:
            count_total += 1
            print('\tget_views thread key: {}. finished until now: {}%'.format(start_key_index, round(
                (float(count_total) / float(images_size) * 100.0), 3)))
            value = image_to_detection_details[key]
            if not value['ret']:
                continue
            key_splitted = key.split('\\')
            cam = key_splitted[7]
            timestamp = float(key.split('\\')[8][0:-4])
            for key_other in image_to_detection_details.keys():
                value_other = image_to_detection_details[key_other]
                key_splitted_other = key_other.split('\\')
                cam_other = key_splitted_other[7]
                if cam == cam_other:
                    continue
                if timestamp in views.keys() and cam_other in views[timestamp]:
                    continue
                timestamp_other = float(key_splitted_other[8][0:-4])
                if value_other['ret']:
                    time_delta = np.abs(timestamp - timestamp_other)
                    if time_delta <= MAX_SECS_BETWEEN_IMAGES:
                        if timestamp not in views.keys():
                            views[timestamp].append(cam)
                        views[timestamp].append(cam_other)

    def get_all_views(self, image_to_detection_details: dict,
                      is_views_from_picke: bool = False):
        print('-' * 80)
        views = None
        views_pickle_file_path = self.calib_folder + '/views.pickle'
        if is_views_from_picke:
            if os.path.exists(views_pickle_file_path):
                print('\tloading views from file..')
                views = pickle.load(open(views_pickle_file_path, "rb"))
            else:
                print('\t', views_pickle_file_path, ' is not exists')
                exit(-1)
        else:
            print('\tcomputing views from scratch..')
            views = defaultdict(list)
            images_size = len(image_to_detection_details)
            keys_indexes_bounds = list(range(0, images_size, int(images_size / VIEWS_THREADS_NUMBER)))
            keys_indexes_bounds[-1] = images_size
            all_keys = list(image_to_detection_details.keys())
            threads_dict = {}
            for i in range(len(keys_indexes_bounds) - 1):
                start_key_index = keys_indexes_bounds[i]
                end_key_index = keys_indexes_bounds[i + 1]
                selected_keys = all_keys[start_key_index:end_key_index]
                threads_dict[start_key_index] = threading.Thread(target=self.get_viwes_from_images_keys,
                                                                 args=(views, image_to_detection_details,
                                                                       selected_keys, start_key_index))
                threads_dict[start_key_index].start()
                print('\tstarted thread of start_index: ', start_key_index)
            for i in range(len(keys_indexes_bounds) - 1):
                start_key_index = keys_indexes_bounds[i]
                threads_dict[start_key_index].join()
                print('\tfinished thread of start_index: ', start_key_index)

            pickle.dump(views, open(views_pickle_file_path, "wb"))
            print('views was saved to file: ', views_pickle_file_path)
        # print('\tviews list {}'.format(views))
        print('-' * 80)
        return views

    @staticmethod
    def build_graph_and_get_ground_cam(all_cams, views, calib_data_json):
        print('-' * 80)
        print('build_graph_and_get_ground_cam')
        # build a graph and calculate ground cam
        g = build_nodes(all_cams)
        create_edges(G=g, views=views)
        ground_cam_suffix = get_center(g)
        print('ground cam: {}'.format(ground_cam_suffix))

        t = nx.bfs_tree(g, source=ground_cam_suffix)

        # show_graph(t, ground_cam_suffix)
        print('-' * 80)
        calib_data_json['graph'] = {}
        calib_data_json['graph']['g_edges'] = list(g.edges)
        calib_data_json['graph']['bfs_tree_edges'] = list(t.edges)
        calib_data_json['graph']['ground_cam_suffix'] = ground_cam_suffix
        return g, t, ground_cam_suffix

    def stereo_calibrate_pair_of_cams(self, i: int, local_gr_suffix: str, other_suffix: str, all_cams: list,
                                      max_imgs: int, calib_data_json: dict, image_to_detection_details: dict,
                                      all_pairs_other_ex_params: dict):
        print('couple {} ({}, {})'.format(i, local_gr_suffix, other_suffix))
        local_gr = get_full_name(all_cams, local_gr_suffix)
        other = get_full_name(all_cams, other_suffix)
        camera_model = self.calibrate_pair_of_cams_inner(local_gr, other, max_imgs, calib_data_json,
                                                         image_to_detection_details)
        pair_key = '{}_{}'.format(local_gr, other)
        R_other = np.round(camera_model['R'], self.float_pre)
        T_other = np.round(camera_model['T'], self.float_pre)

        other_ex_params = {
            # 'camera_model': camera_model_lists,
            'rotation_mat': R_other.tolist(),
            'translation': T_other.tolist()
        }
        all_pairs_other_ex_params[pair_key] = other_ex_params
        print('\tcam path {}: relative ex params: {}'.format(other, utils.json_to_string(other_ex_params)))

    def calc_relative_extrinsic_params(self, t: nx.Graph, all_cams: list, max_imgs: int, calib_data_json: dict,
                                       image_to_detection_details: dict, is_all_relative_pairs_from_pickle: bool):
        print('-' * 80)
        print('calc_relative_extrinsic_params(max {} images needed)'.format(max_imgs))
        all_pairs_other_ex_params = None

        all_relative_pairs_pickle_file_path = self.calib_folder + '/all_relative_pairs.pickle'
        if is_all_relative_pairs_from_pickle:
            if os.path.exists(all_relative_pairs_pickle_file_path):
                print('\tloading all_relative_pairs from file..')
                all_pairs_other_ex_params = pickle.load(open(all_relative_pairs_pickle_file_path, "rb"))
            else:
                print('\t', all_relative_pairs_pickle_file_path, ' is not exists')
                exit(-1)
        else:
            print('\tcomputing all_relative_pairs from scratch..')
            all_pairs_other_ex_params = {}
            t_edges = list(nx.bfs_edges(t, source=self.ground_cam_suffix))
            threads_dict = {}
            for i, (local_gr_suffix, other_suffix) in enumerate(t_edges):
                thread_id = local_gr_suffix + '_' + other_suffix
                threads_dict[thread_id] = threading.Thread(target=self.stereo_calibrate_pair_of_cams,
                                                           args=(i, local_gr_suffix, other_suffix,
                                                                 all_cams, max_imgs, calib_data_json,
                                                                 image_to_detection_details,
                                                                 all_pairs_other_ex_params))
                threads_dict[thread_id].start()
                print('\tstarted thread of start_index: ', thread_id)
            for i, (local_gr_suffix, other_suffix) in enumerate(t_edges):
                thread_id = local_gr_suffix + '_' + other_suffix
                threads_dict[thread_id].join()
                print('\tfinished thread: ', thread_id)

            pickle.dump(all_pairs_other_ex_params, open(all_relative_pairs_pickle_file_path, "wb"))
            print('all_relative_pairs was saved to file: ', all_relative_pairs_pickle_file_path)
        print('-' * 80)
        return all_pairs_other_ex_params

    def calc_real_extrinsic_params(self, calib_data_json: dict, t: nx.Graph, all_cams: list,
                                   all_pairs_other_ex_params: dict):
        print('-' * 80)
        print('calc_real_extrinsic_params:')
        # ground mac rotation and position are fixed
        ground_cam_mac = self.ground_cam.split('/')[0]
        ground_cam_port = self.ground_cam.split('/')[1]
        R_ground = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        p_ground = np.array([0.0, 0.0, 0.0], dtype=float)
        v_ground = utils.get_3d_orientation_from_2d_point(x=0, y=0, fx=1, R=R_ground, float_pre=self.float_pre)
        ground_cam_extrinsic_params = {
            # 'camera_model': None,
            'position': p_ground.tolist(),
            'orientation': v_ground.tolist(),
            'rotation_mat': R_ground.tolist(),
        }
        calib_data_json[ground_cam_mac][ground_cam_port].update(ground_cam_extrinsic_params)
        print('\tcam path {} (GROUND CAM). final ex params {}'.format(
            self.ground_cam_suffix, utils.json_to_string(ground_cam_extrinsic_params)))

        all_paths_edges = nx.shortest_path(G=t, source=self.ground_cam_suffix)
        print('all_paths_edges: {}'.format(all_paths_edges))

        for other_suffix, path_from_gr in all_paths_edges.items():
            if len(path_from_gr) >= 3:
                x = 2
            if len(path_from_gr) >= 2:
                print('handling ex params of {}'.format(other_suffix))
                other = get_full_name(all_cams, other_suffix)
                other_mac, other_port = other.split('/')[0], other.split('/')[1]
                R_1_k, p_1_k = np.eye(utils.DIM_3), np.zeros(
                    utils.DIM_3)  # Rotation and position relative to global ground cam
                path_from_gr_rev = path_from_gr[:]
                path_from_gr_rev.reverse()
                for i in range(len(path_from_gr_rev) - 1):
                    local_gr_suffix, local_other_suffix = path_from_gr_rev[i + 1], path_from_gr_rev[i]
                    local_gr = get_full_name(all_cams, local_gr_suffix)
                    local_other = get_full_name(all_cams, local_other_suffix)
                    pair_key = '{}_{}'.format(local_gr, local_other)
                    # pair_key = '{}_{}'.format(local_other, local_gr) # for reverse order
                    print('\t{} pair_key {}'.format(i, pair_key))
                    other_ex_params = all_pairs_other_ex_params[pair_key]
                    R_i_i_plus_1 = np.array(other_ex_params['rotation_mat'], dtype=float)
                    T_i_i_plus_1 = np.array(other_ex_params['translation'], dtype=float)
                    p_i_i_plus_1 = utils.get_3d_position_from_R_and_T(
                        R=R_i_i_plus_1,
                        T=T_i_i_plus_1,
                        float_pre=self.float_pre
                    )
                    p_1_k = (p_1_k - p_i_i_plus_1).T @ R_i_i_plus_1.T
                    R_1_k = R_i_i_plus_1.T @ R_1_k

                v_other = utils.get_3d_orientation_from_2d_point(x=0, y=0, fx=1, R=R_1_k, float_pre=self.float_pre)
                other_ex_params = {
                    'position': np.round(p_1_k.tolist(), self.float_pre).tolist(),
                    'orientation': np.round(v_other, self.float_pre).tolist(),
                    'rotation_mat': np.round(R_1_k, self.float_pre).tolist()
                }
                calib_data_json[other_mac][other_port].update(other_ex_params)
                print(
                    '\tcam path {}. final ex params {}'.format(other_suffix, utils.json_to_string(other_ex_params)))
        print('-' * 80)
        return

    def calibrate_pair_of_cams_inner(self, ground_cam_path: str, other_cam_path: str, max_imgs: int,
                                     calib_data_json: dict,
                                     image_to_detection_details: dict) -> dict:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        obj_p = np.zeros((self.squares_num_width * self.squares_num_height, 3), np.float32)
        obj_p[:, :2] = np.mgrid[0:self.squares_num_width, 0:self.squares_num_height].T.reshape(-1, 2) * self.square_size

        ground_cam_mac, ground_cam_port = ground_cam_path.split('/')
        other_cam_mac, other_cam_port = other_cam_path.split('/')
        ground_cam_images, other_cam_images = [], []
        for key, value in image_to_detection_details.items():
            current_mac = value['mac']
            current_cam_port = value['cam_port']
            if current_mac == ground_cam_mac and current_cam_port == ground_cam_port:
                other_cam_images.append(key)
            elif current_mac == other_cam_mac and current_cam_port == other_cam_port:
                ground_cam_images.append(key)

        images_joint = []
        for image in ground_cam_images:
            timestamp = float(image.split('\\')[8][0:-4])
            for image_other in other_cam_images:
                timestamp_other = float(image_other.split('\\')[8][0:-4])
                time_delta = np.abs(timestamp - timestamp_other)
                if time_delta <= MAX_SECS_BETWEEN_IMAGES:
                    images_joint.append([image, image_other])
        # print('\timages_joint(exist on both folders): {}'.format(images_joint))
        assert len(images_joint) > 0, 'pair ({},{}) no images found with the same name'.format(ground_cam_path,
                                                                                               other_cam_path)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img1_points = []  # 2d points in image plane.
        img2_points = []  # 2d points in image plane.
        img_shape = None
        pass_both_macs = 0
        imgs_checked = 0
        random.shuffle(images_joint)
        for img_1_path, img_2_path in images_joint:
            imgs_checked += 1

            # If found, add object points, image points (after refining them)
            if image_to_detection_details[img_1_path]['ret'] and image_to_detection_details[img_2_path]['ret']:
                if img_shape is None:
                    img_shape = image_to_detection_details[img_1_path]['image_shape']
                pass_both_macs += 1
                obj_points.append(obj_p)

                img1_points.append(image_to_detection_details[img_1_path]['corners2'])

                img2_points.append(image_to_detection_details[img_2_path]['corners2'])

                if self.show_corners:  # Draw and display the corners
                    cv2.drawChessboardCorners(image_to_detection_details[img_1_path]['image'],
                                              self.squares_num_w_and_h,
                                              image_to_detection_details[img_1_path]['corners2'],
                                              image_to_detection_details[img_1_path]['ret'])
                    cv2.drawChessboardCorners(image_to_detection_details[img_2_path]['image'],
                                              self.squares_num_w_and_h,
                                              image_to_detection_details[img_2_path]['corners2'],
                                              image_to_detection_details[img_2_path]['ret'])
                    cv2.imshow('img1={}, img2={}'.format(img_1_path, img_2_path),
                               np.concatenate((image_to_detection_details[img_1_path]['image'],
                                               image_to_detection_details[img_2_path]['image']), axis=1))
                    cv2.waitKey()
            if pass_both_macs >= max_imgs:
                break
        print('\t\tpair: [', ground_cam_path, ', ', other_cam_path,
              ']. pass_both_macs={}, checked {}/{} images'.format(pass_both_macs, imgs_checked, len(images_joint)))
        assert pass_both_macs > 0, 'pair ({},{}) must detect chessboard on at least one joint image'.format(
            ground_cam_path, other_cam_path)

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        # this outputs the intrinsic matrix of each one of the two cameras, and all the
        #  rotation and translation from first cam to second cam
        ground_cam_mac, ground_cam_port = ground_cam_path.split('/')
        other_cam_mac, other_cam_port = ground_cam_path.split('/')
        stereo_calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.00001)

        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img1_points,
            img2_points,
            np.asarray(calib_data_json['calib_data'][ground_cam_mac][ground_cam_port]['intrinsic_mat']),
            np.asarray(calib_data_json['calib_data'][ground_cam_mac][ground_cam_port]['distortion']),
            np.asarray(calib_data_json['calib_data'][other_cam_mac][other_cam_port]['intrinsic_mat']),
            np.asarray(calib_data_json['calib_data'][other_cam_mac][other_cam_port]['distortion']),
            img_shape,
            criteria=stereo_calib_criteria,
            flags=flags)
        camera_model = {
            'M1': np.round(M1, self.float_pre),
            'M2': np.round(M2, self.float_pre),
            'dist1': np.round(d1, self.float_pre),
            'dist2': np.round(d2, self.float_pre),
            # 'rvecs1': np.round(np.asarray(calib_data_json['calib_data'][ground_cam_mac][ground_cam_port]['r']),
            #                    self.float_pre),
            # 'rvecs2': np.round(np.asarray(calib_data_json['calib_data'][other_cam_mac][other_cam_port]['r']),
            #                    self.float_pre),
            'R': np.round(R, self.float_pre),
            'T': np.round(T, self.float_pre),
            'E': np.round(E, self.float_pre),
            'F': np.round(F, self.float_pre)
        }
        # print('ground_cam_path:\n', ground_cam_path)
        # print('other_cam_path:\n', other_cam_path)
        # print('camera_model:\n', camera_model)
        cv2.destroyAllWindows()
        return camera_model

    def initialize(self):
        print('-' * 80)
        print('initialize:')
        all_dirs_names = os.listdir(self.calib_images_folder)
        print('\tfound in calib_images_folder directories: {}'.format(all_dirs_names))

        all_cams = []  # each cam is a tuple of rp mac and port
        for mac_dir in all_dirs_names:
            ports_folders = os.listdir('{}/{}'.format(self.calib_images_folder, mac_dir))
            for port_str in ports_folders:
                cam_mac_port = '{}/{}'.format(mac_dir, port_str)
                all_cams.append(cam_mac_port)
        print('\tall_cams: {}'.format(all_cams))

        calib_data_json = {'calib_data': {}}
        for cam_path in all_cams:
            mac_str = cam_path.split('/')[0]
            cam_port_str = cam_path.split('/')[1]
            if mac_str not in calib_data_json['calib_data']:
                calib_data_json['calib_data'][mac_str] = {}
            calib_data_json['calib_data'][mac_str][cam_port_str] = {}
        print('-' * 80)
        return calib_data_json, all_cams, all_dirs_names

    def finalize_and_save(self, all_cams, all_dirs_names, calib_data_json):
        calib_data_json_upper = {'calib_data': {}, 'graph': {}}
        for calib_key in ['calib_data', 'graph']:
            for k, v in calib_data_json[calib_key].items():
                fixed_key = None
                if calib_key == 'calib_data':
                    upper = k.upper()
                    fixed_key = ':'.join(
                        upper[i:min(i + MAC_STR_JUMP, MAC_STR_SIZE)] for i in range(0, MAC_STR_SIZE, MAC_STR_JUMP))
                elif calib_key == 'graph':
                    fixed_key = k
                calib_data_json_upper[calib_key][fixed_key] = v

        date_for_folder = utils.get_time_stamp_for_folder_name()  # add time stamp YYYY_MM_DD
        num_devices = len(all_dirs_names)  # add number of rps
        num_cams = len(all_cams)  # add number of cams
        glob_mac_1 = os.path.join(self.calib_images_folder, self.ground_cam, '*.jpg')
        images = glob.glob(glob_mac_1)
        num_images_per_cam = len(images)  # add number of images per cam
        new_suffix = 'calib_file_{}_{}devices_{}cams_{}images.json'.format(date_for_folder, num_devices, num_cams,
                                                                           num_images_per_cam)
        # saving one folder above CalibImages
        calib_file_path = os.path.join(self.calib_folder, new_suffix)
        print('preparing calib file {}'.format(calib_file_path))

        print('Saving Calibration output to {}'.format(calib_file_path))
        utils.save_json(calib_file_path, calib_data_json_upper, ack=False)
        return

    def plot_and_graphs(self, g, bfs_tree, calib_data_json):
        if self.show_views_graph:
            fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
            utils.move_plot(fig, where='top_right')
            add_graph(g, ax=axes[0], center_node=self.ground_cam_suffix, title='full_graph')
            add_graph(bfs_tree, ax=axes[1], center_node=self.ground_cam_suffix, title='MST')
        if self.show_plot:
            cameras_params = utils.CameraParams.read_calib_data(calib_data_json['calib_data'], rps_playing=None)
            self.build_plot(location='top_left')  # build empty 3d plot
            colors_map = utils.get_color_map(n=len(cameras_params))
            for i, camera_params in enumerate(cameras_params):
                self.update_plot(cam_ind=i, camera_params=camera_params, c=colors_map[i])
            utils.add_orientation_arrows_to_points(self.axes, cameras_params, 'r', 100)

            utils.add_orientation_pyramids_to_cams_pyplot(
                self.axes,
                cameras_params,
                color="orange",
                dist=3,
                height=400,
                width=400)

            print('Waiting for you to close the plot')
        plt.show(block=True)
        return

    def run_calibration(self):
        calib_data_json, all_cams, all_dirs_names = self.initialize()

        image_to_detection_details = self.get_images_detection_details(IS_IMAGES_TO_DETECTION_FROM_PICKLE,
                                                                       all_cams)

        calib_data_json = self.calc_intrinsic_params(calib_data_json, image_to_detection_details, all_cams,
                                                     INTRINSIC_MAX_IMGS, IS_CALIB_DATA_FROM_PICKLE)

        if CALIB_MODE == INT:
            return

        views = self.get_all_views(image_to_detection_details, IS_VIEWS_FROM_PICKLE)
        # views = temp_get_views()  # todo remove

        g, bfs_tree, self.ground_cam_suffix = self.build_graph_and_get_ground_cam(all_cams, views,
                                                                                  calib_data_json)
        # self.plot_and_graphs(g, bfs_tree, calib_data_json)

        # g, bfs_tree, self.ground_cam_suffix = temp_build_graph_and_set_ground_cam(all_cams, views)  # todo remove
        self.ground_cam = get_full_name(all_cams, self.ground_cam_suffix)

        # Third calculate relative extrinsic params(relative to "local" ground cam)
        all_pairs_other_ex_params = self.calc_relative_extrinsic_params(bfs_tree, all_cams,
                                                                        RELATIVE_EXTRINSIC_SINGLE_PAIR_MAX_IMAGES,
                                                                        calib_data_json, image_to_detection_details,
                                                                        IS_ALL_RELATIVE_PAIRS_FROM_PICKLE)  # todo uncomment
        # all_pairs_other_ex_params = temp_calc_relative_extrinsic_params()  # todo remove

        # Fourth calculate real extrinsic params (relative to ground cam)
        self.calc_real_extrinsic_params(calib_data_json['calib_data'], bfs_tree, all_cams,
                                        all_pairs_other_ex_params)

        # Finalize
        self.plot_and_graphs(g, bfs_tree, calib_data_json)
        self.finalize_and_save(all_cams, all_dirs_names, calib_data_json)
        return


def add_graph(G: nx.Graph, ax, center_node: str = None, title: str = '') -> None:
    ax.set_title(title)
    pos = nx.spring_layout(G)
    if center_node is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=set(G.nodes - [center_node]), ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[center_node], node_color='r', ax=ax)
    else:
        nx.draw(G, pos, ax=ax)

    nx.draw_networkx_labels(G, pos, ax=ax)
    return


def build_nodes(cams: list) -> nx.Graph:
    G = nx.Graph()
    for cam in cams:
        G.add_node(cam[-MAC_PORT_MIN_CHARS:])
    return G


def create_edges(G: nx.Graph, views: dict) -> None:
    edge_to_counter = defaultdict(int)
    for timestamp, cams in views.items():
        n_choose_2 = list(itertools.combinations(cams, 2))

        for cam1, cam2 in n_choose_2:
            e = (cam1[-MAC_PORT_MIN_CHARS:], cam2[-MAC_PORT_MIN_CHARS:])
            edge_key = str(e)
            edge_to_counter[edge_key] += 1
            if edge_to_counter[edge_key] > MIN_IMAGES_FOR_EDGE and not G.has_edge(*e):
                G.add_edge(*e)
    return


def get_center(G):
    """
    if more than one center found, get the highest degree one
    :param G:
    :return:
    """

    print(G.adj)
    centers = center(G)  # could be more than 1
    best_center = None
    print('\tlooking for best center (maximal out degree)')
    if len(centers) > 1:
        max_degree = -1
        for c in centers:
            c_degree = len(list(G.neighbors(c)))
            print('\t\t{} has {} neighbours'.format(c, c_degree))
            if c_degree > max_degree:
                # print('\t new best {}'.format(c))
                max_degree = c_degree
                best_center = c
    else:
        best_center = centers[0]
    c_degree = len(list(G.neighbors(best_center)))
    print('\t\tselected center {} has {} neighbours'.format(best_center, c_degree))
    return best_center
    # return 'F9/1'


def get_full_name(all_cams: list, suffix: str) -> str:
    for cam in all_cams:
        if cam.endswith(suffix):
            return cam
    return None


def main():
    print("Python  Version {}".format(sys.version))
    print('Working dir: {}'.format(os.getcwd()))
    PROJECT_HOME = '../../../'  # relative to ServerMain.py
    PROJECT_ABS_PATH = os.path.abspath(PROJECT_HOME)
    print('PROJECT_ABS_PATH: {}'.format(PROJECT_ABS_PATH))
    os.chdir(PROJECT_ABS_PATH)

    CFG_GLOBALS = utils.load_json('SharedCode/GlobalConstants.json', ack=False)
    CFG_CALIB = utils.load_json('Server/ServerConstants.json', ack=False)['calibration']['part2']
    CALIB_FOLDER = os.path.join(PROJECT_ABS_PATH, 'SharedCode/Calib', CFG_GLOBALS['misc_global']['calib_folder'])

    calib_agent = CalibAgent(
        calib_folder=CALIB_FOLDER,
        float_pre=CFG_GLOBALS['misc_global']['float_precision'],
        show_corners=CFG_CALIB['show_corners'],
        show_plot=CFG_CALIB['show_plot'],
        show_views_graph=CFG_CALIB['show_views_graph'],
        square_size=CFG_CALIB['square_size'],
        squares_num_width=CFG_CALIB['squares_num_width'],
        squares_num_height=CFG_CALIB['squares_num_height']
    )
    print(calib_agent)
    calib_agent.run_calibration()
    return


if __name__ == '__main__':
    main()

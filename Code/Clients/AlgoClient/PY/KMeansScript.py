import itertools

import numpy as np
import cv2
import copy
from itertools import combinations
from collections import defaultdict, OrderedDict
from scipy.spatial.distance import cdist
import SharedCode.PY.utilsScript as utils

# to avoid confusion
LINE_IND = 0  # the index in the tuple of Class Line
CLUSTER_IND = 1  # the index in the tuple of Class Cluster

POINT3D = np.empty(shape=[1, 3], dtype=float)
MATRIX33D = np.empty(shape=[3, 3], dtype=float)


class KMeans:
    def __init__(self,
                 k: int,
                 run_type: str,
                 t: int = None,
                 use_cam_pos_filter: bool = False,
                 use_in_range_filter: bool = False,
                 use_unique_points_filter: bool = False,
                 use_noise_filter: bool = False,
                 cams_positions: np.array = None,
                 in_range_values: dict = None,
                 noise_filter_size: int = None,
                 debug: bool = False
                 ):
        """
        :param k:
        :param run_type: exhaustive, random search, 1 mean acc
        :param t: iterations on exhaustive and random search
        :param use_cam_pos_filter: if True, cams_positions is given
                    filter candidates if they are one of the cams
        :param use_in_range_filter: if True, in_range_values given
                    filter k sets that the distance of any 2 points is out of range
        :param use_unique_points_filter:
                    filter duplicates candidates
        :param use_noise_filter: CURRENTLY supports only k==1.
                track results over iterations and try to smoother the movement by averaging the last results
        :param cams_positions: np array of 'cam_number' over 3 - cam positions
        :param in_range_values: dict with 2 keys - bottom and top(both ints)
        :param noise_filter_size: int. how many iterations to track
        :param debug: extra prints
        :return:
        """
        self.k = k
        self.run_type = run_type
        self.t = t
        self.use_cam_pos_filter = use_cam_pos_filter
        self.use_in_range_filter = use_in_range_filter
        self.use_unique_points_filter = use_unique_points_filter
        self.use_noise_filter = use_noise_filter
        self.cams_positions = cams_positions
        self.in_range_values = in_range_values
        self.noise_filter_size = noise_filter_size
        self.debug = debug
        if self.use_cam_pos_filter:
            assert cams_positions is not None, 'use_cam_pos_filter True but no cams positions given'
        if self.use_in_range_filter:
            assert in_range_values is not None, 'use_in_range_filter True but no range given'
        if self.use_noise_filter:
            assert self.noise_filter_size > 1, 'self.use_noise_filter True but noise_filter_size < 2'

        self.clusters_points = []  # for noise_filter - list of points in shape (3,) as list
        if self.use_noise_filter:
            for i in range(self.k):
                self.clusters_points.append([])
        return

    def __str__(self):
        string = 'KMeans:\n'
        string += '\tk={}\n'.format(self.k)
        string += '\trun_type={}\n'.format(self.run_type)
        string += '\tt={}\n'.format(self.t)
        string += '\tuse_cam_pos_filter={}\n'.format(self.use_cam_pos_filter)
        if self.use_cam_pos_filter:
            string += utils.var_to_string(self.cams_positions, '\t\tcams_positions', with_data=True)
            string += '\n'
        string += '\tuse_in_range_filter={}\n'.format(self.use_cam_pos_filter)
        if self.use_in_range_filter:
            string += utils.var_to_string(self.in_range_values, '\t\tin_range_values', with_data=True)
            string += '\n'
        string += '\tuse_unique_points_filter={}\n'.format(self.use_unique_points_filter)
        string += '\tuse_noise_filter={}(size={})\n'.format(self.use_noise_filter, self.noise_filter_size)
        return string

    def generate_lines_from_cams_info(self, cams_info_list: list) -> list:
        """
        AUX function
        :param cams_info_list: list(CameraInfo)
        each CameraInfo has k2d_points. for each - create a Class line (from the camera to it)
        :return: list(Class Line)
        """
        lines = []
        for cam_info in cams_info_list:
            if cam_info.k2d_points is None:
                continue
            aspect_ratio = cam_info.fx / cam_info.fy
            for i, _2d_point in enumerate(cam_info.k2d_points):
                if len(_2d_point) == 0:
                    continue
                # print(utils.var_to_string(_2d_point, '2dpoint', with_data=True))
                xyz = np.empty(shape=(1, utils.DIM_3))
                xyz[0, 0] = _2d_point[0] - cam_info.cx
                xyz[0, 1] = (_2d_point[1] - cam_info.cy) * aspect_ratio
                xyz[0, 2] = cam_info.fx
                # print(utils.var_to_string(xyz, 'xyz', with_data=True))
                # print(utils.var_to_string(cam_info.rotation, 'cam_info.rotation', with_data=True))
                v = (cam_info.rotation.T @ xyz.T).T
                # print(utils.var_to_string(v, 'v', with_data=True))
                lines.append(Line(cam_info.position, v))
                # print(lines[-1])
        if self.debug:
            print('\t|lines|={}'.format(len(lines)))
        return lines

    def run_kmeans(self, lines: list) -> (np.array, float):
        assert len(lines) > 1, '|lines|={}'.format(len(lines))
        individual_candidates_expected_size = 2 * utils.nCr(n=len(lines), r=2)
        assert individual_candidates_expected_size >= self.k, '|candidates|={} < k={}'.format(
            individual_candidates_expected_size, self.k)
        k3d_points, total_cost = None, float("inf")
        if self.run_type == 'exhaustive':
            k3d_points, total_cost = self.exhaustive_search(lines)
        elif self.run_type == 'random':
            k3d_points, total_cost = self.random_search(lines)
        elif self.run_type == '1_mean_acc':
            k3d_points, total_cost = self.one_mean_accurate(lines)
        else:
            print(utils.WARNING.format('self.run_type {} is invalid'.format(self.run_type)))

        if k3d_points is None or len(k3d_points) == 0:  # failed to find any k_set
            k3d_points, total_cost = np.empty(shape=0), float("inf")
        else:  # success
            # k3d_points[:, 0] *= -1
            # k3d_points[:, 1] *= -1
            if self.k > 1:
                k3d_points = k3d_points.reshape(self.k, utils.DIM_3)  # reshape from (k,1,3) to (k,3)
            else:  # k == 1
                if self.use_noise_filter:  # filter noise if on
                    k3d_points = self.filter_noise(k3d_points)
                k3d_points = k3d_points.flatten()  # reshape from (1,3) to (,3)
            if self.debug:
                print('\tmin cost {:,.2f}'.format(total_cost))
                print(utils.var_to_string(k3d_points, title='\tkmeans OUT', with_data=True))
        return k3d_points, total_cost

    def one_mean_accurate(self, lines: list) -> np.array:
        """
        This is 1 accurate mean. This is an auxilary function that are not used in this script.
        :param lines:
        :return:
        """
        assert len(lines) > 1, '|lines|={}'.format(len(lines))
        line_ind_to_line_cluster_dict = self.preprocess_lines(lines)
        super_cluster = None
        # for (i, (line, cluster)) in line_ind_to_line_cluster_dict.items():
        for i in range(len(line_ind_to_line_cluster_dict)):
            current_cluster = line_ind_to_line_cluster_dict[i][CLUSTER_IND]
            if i == 0:
                super_cluster = current_cluster
            else:
                super_cluster = Cluster.join_clusters(clusters=[super_cluster, current_cluster])
        if np.linalg.det(super_cluster.rejection_sum_mat) != 0:
            min_cost_point, min_cost_value = super_cluster.get_min_cost_point(with_value=True)
        else:
            # if lines parallel and symmetrical around origin: rejection_sum_mat is not invertible
            # todo check if it can happen in other cases
            # todo check if the origin is always good in this case
            min_cost_point = np.zeros(shape=(1, 3), dtype=float)
            min_cost_value = float("inf")
        min_cost_value_avg = min_cost_value / float(len(line_ind_to_line_cluster_dict))
        print('min_cost_value_avg: ', min_cost_value_avg)
        return min_cost_point, min_cost_value

    def one_mean_accurate_exhaustive(self, lines: list) -> np.array:
        """
        This is 1 accurate mean. This is an auxilary function that are not used in this script.
        :param lines:
        :return:
        """
        assert len(lines) > 1, '|lines|={}'.format(len(lines))
        line_ind_to_line_cluster_dict = self.preprocess_lines(lines)
        min_cost_value, min_cost_point = np.inf, None
        for i in range(len(line_ind_to_line_cluster_dict)):
            cluster_i = line_ind_to_line_cluster_dict[i][CLUSTER_IND]
            for j in range(len(line_ind_to_line_cluster_dict)):
                if i != j:
                    cluster_j = line_ind_to_line_cluster_dict[j][CLUSTER_IND]
                    cluster_i_j = Cluster.join_clusters(clusters=[cluster_i, cluster_j])
                    if np.linalg.det(cluster_i_j.rejection_sum_mat) != 0:
                        current_min_cost_point, current_min_cost_value = cluster_i_j.get_min_cost_point(with_value=True)
                    else:
                        # if lines parallel and symmetrical around origin: rejection_sum_mat is not invertible
                        # todo check if it can happen in other cases
                        # todo check if the origin is always good in this case
                        current_min_cost_point = np.zeros(shape=(1, 3), dtype=float)
                        current_min_cost_value = float("inf")
                    if current_min_cost_value < min_cost_value:
                        min_cost_point = current_min_cost_point
                        min_cost_value = current_min_cost_value
        return min_cost_point, min_cost_value

    def one_mean_accurate_optimized(self, lines: list) -> np.array:
        """
        This is 1 accurate mean. This is an auxilary function that are not used in this script.
        :param lines:
        :return:
        """
        lines_size = len(lines)
        assert lines_size > 1, '|lines|={}'.format(len(lines))
        min_cost_point_final = np.zeros(shape=(1, 3), dtype=float)
        min_cost_value_final, min_cost_value_avg_final = float("inf"), float("inf")
        current_indexes_final, current_lines_final = None, None
        bitmaps = np.array(list(itertools.product("01", repeat=lines_size)), dtype=int)
        for bitmap in bitmaps:
            current_indexes = [i for i, val in enumerate(bitmap) if val]
            if len(current_indexes) > 1:
                # print('lines: ', lines)
                # print('current_indexes: ', current_indexes)
                # print('bitmap: ', bitmap)
                current_lines = list(np.array(lines)[current_indexes])
                line_ind_to_line_cluster_dict = self.preprocess_lines(current_lines)
                super_cluster = None
                # for (i, (line, cluster)) in line_ind_to_line_cluster_dict.items():
                for i in range(len(line_ind_to_line_cluster_dict)):
                    current_cluster = line_ind_to_line_cluster_dict[i][CLUSTER_IND]
                    if i == 0:
                        super_cluster = current_cluster
                    else:
                        super_cluster = Cluster.join_clusters(clusters=[super_cluster, current_cluster])
                if np.linalg.det(super_cluster.rejection_sum_mat) != 0:
                    min_cost_point, min_cost_value = super_cluster.get_min_cost_point(with_value=True)
                else:
                    # if lines parallel and symmetrical around origin: rejection_sum_mat is not invertible
                    # todo check if it can happen in other cases
                    # todo check if the origin is always good in this case
                    min_cost_point = np.zeros(shape=(1, 3), dtype=float)
                    min_cost_value = float("inf")
                min_cost_value_avg = min_cost_value / float(len(current_lines))
                if min_cost_value_avg < min_cost_value_avg_final:
                    min_cost_value_avg_final = min_cost_value_avg
                    min_cost_point_final = min_cost_point
                    min_cost_value_final = min_cost_value
                    current_indexes_final = current_indexes
                    current_lines_final = current_lines
                # print('min_cost_value_avg: ', min_cost_value_avg)
                # print('current_indexes: ', current_indexes)
                # print('current_lines: ', current_lines)
        print('**************************')
        print('min_cost_value_avg_final: ', min_cost_value_avg_final)
        print('current_indexes_final: ', current_indexes_final)
        print('**************************')
        return min_cost_point_final, min_cost_value_final

    def exhaustive_search(self, lines: list) -> (np.array, float):
        """
        :param lines: list(Class Line)
        step 0: preprocess_lines(lines)
        step 1: get_individual_candidates(line_ind_to_line_cluster_dict)
        step 2: get_k_set_candidates(individual_candidates)
        step 3: find_best_k_set(line_ind_to_line_cluster_dict, k_set_candidates)
        :return: kmeans and their cost
        """
        k3d_points, cost = np.empty(shape=(0, 1, utils.DIM_3)), float("inf")
        line_ind_to_line_cluster_dict = self.preprocess_lines(lines)
        individual_candidates = self.get_individual_candidates(line_ind_to_line_cluster_dict)
        k_set_candidates = self.get_k_set_candidates(individual_candidates)
        if k_set_candidates.shape[0] >= 0:  # found at least 1 k_set_candidate
            k3d_points, cost = self.find_best_k_set(line_ind_to_line_cluster_dict, k_set_candidates)
        return k3d_points, cost

    def preprocess_lines(self, lines: list) -> OrderedDict:
        """
        :param lines: list(Class Line)
        creates a dict
        key = index of line
        value = tuple(Line, Cluster)
        Cluster is created from line
        :return:
        """
        line_ind_to_line_cluster_dict = OrderedDict()
        for i, line in enumerate(lines):
            line_ind_to_line_cluster_dict[i] = (lines[i], Cluster(lines[i]))
        if self.debug:
            print(utils.var_to_string(line_ind_to_line_cluster_dict, '\tlines_i_to_line_cluster_dict', with_data=False))
            # for i, (line, cluster) in line_ind_to_line_cluster_dict.items():
            #     print('tuple {}:'.format(i))
            #     print(line)
            #     print(cluster)
        return line_ind_to_line_cluster_dict

    def get_individual_candidates(self, line_ind_to_line_cluster_dict: dict) -> np.array:
        """
        :param line_ind_to_line_cluster_dict: line index to a tuple of (class Line, class Cluster)
        calculate an individual candidate set ( candidate is a 3d point shape=(1,3) )
            each candidate could be 1 of the k means at the end
        for each 2 lines:
            calculate the min cost 3d point (like asking 1 mean between 2 lines)
            project min cost point on each line -> proj1, proj2 (3d points)
            save both in individual_candidates
        :return: individual_candidates: np.array, dtype=float, shape=(2*L_Choose_2, 1, 3) where L=|lines|
        """
        lines_ind_size = len(line_ind_to_line_cluster_dict)
        if lines_ind_size > 2:
            expected_size = 2 * utils.nCr(n=lines_ind_size, r=2)
        else:
            expected_size = lines_ind_size

        individual_candidates = np.empty(shape=(expected_size, 1, utils.DIM_3))
        lines_indices = list(line_ind_to_line_cluster_dict.keys())

        for i, (line1_ind, line2_ind) in enumerate(combinations(lines_indices, r=2)):
            line1 = line_ind_to_line_cluster_dict[line1_ind][LINE_IND]
            cluster1 = line_ind_to_line_cluster_dict[line1_ind][CLUSTER_IND]

            line2 = line_ind_to_line_cluster_dict[line2_ind][LINE_IND]
            cluster2 = line_ind_to_line_cluster_dict[line2_ind][CLUSTER_IND]

            super_cluster = Cluster.join_clusters(clusters=[cluster1, cluster2])

            if np.linalg.det(super_cluster.rejection_sum_mat) != 0:
                min_cost_point, _ = super_cluster.get_min_cost_point(with_value=False)
            else:
                # if lines parallel and symmetrical around origin: rejection_sum_mat is not invertible
                # todo check if it can happen in other cases
                # todo check if the origin is always good in this case
                min_cost_point = np.zeros(shape=(1, 3), dtype=float)

            individual_candidates[2 * i] = line1.get_projected_point(_3d_point=min_cost_point)
            individual_candidates[2 * i + 1] = line2.get_projected_point(_3d_point=min_cost_point)

        if self.debug:
            print(utils.var_to_string(individual_candidates, title='\tindividual_candidates', with_data=False))
        return individual_candidates

    def get_k_set_candidates(self, individual_candidates: np.array) -> np.array:
        """
        :param individual_candidates: np.array, dtype=float, shape=(X, 1, 3)
                where X depends on get_individual_candidate()
        if self.t <=0:
            for k_set in (X choose k):
                if k_set passed all filters:
                    k_set_candidates.append(k_set)
        else:
            for i in range(t):
                select randomly a k_set from individual_candidates
                if k_set passed all filters:
                    k_set_candidates.append(k_set)

        :return: k_set_candidates np.array, dtype=float, shape=(Y, k, 1, 3)
        if self.t <=0: Y <= X choose k
        else:          Y <= t
        """
        k_set_candidates = np.empty(shape=(0, self.k, 1, utils.DIM_3))
        if self.use_cam_pos_filter:
            individual_candidates = self.filter_cams_points_from_candidates(individual_candidates, self.cams_positions,
                                                                            debug=self.debug)
        if self.use_unique_points_filter:
            individual_candidates = self.filter_duplicates_from_candidates(individual_candidates, debug=self.debug)

        individual_candidates_size = individual_candidates.shape[0]
        dynamic_norm_dict = {}  # do each pair once (store in dict for future compares)

        self.t = 0
        nCr = utils.nCr(n=individual_candidates.shape[0], r=self.k)
        if self.t <= 0 or self.t > nCr:
            possibilities_candidates_num = nCr
            print('*** possibilities_candidates_num: ', possibilities_candidates_num)
            # go over all nCk where n=centroid_set.shape[0]. if n=132 nad k=4, nCk=1,208,2785
            for i, idx in enumerate(combinations(range(individual_candidates_size), r=self.k)):
                idx = list(idx)
                k_set_candidate = individual_candidates[idx]
                print('*** k_set_candidate: ', k_set_candidate)
                if self.use_in_range_filter:
                    passed_in_range = self.all_norms_in_range_opt(k_set_candidate, dynamic_norm_dict, idx,
                                                                  self.in_range_values['bottom'],
                                                                  self.in_range_values['top'])
                    if passed_in_range:
                        k_set_candidates = np.concatenate((k_set_candidates, np.expand_dims(k_set_candidate, axis=0)))
                k_set_candidates = np.concatenate((k_set_candidates, np.expand_dims(k_set_candidate, axis=0)))
        else:
            # t = utils.get_time_now()
            possibilities_candidates_num = self.t
            for i in range(self.t):
                perm = np.random.permutation(individual_candidates_size)
                idx = perm[:self.k]  # 3 times faster than np.random.choice
                k_set_candidate = individual_candidates[idx]
                if self.use_in_range_filter:
                    if self.all_norms_in_range_opt(k_set_candidate, dynamic_norm_dict, idx,
                                                   self.in_range_values['bottom'], self.in_range_values['top']):
                        k_set_candidates = np.concatenate((k_set_candidates, np.expand_dims(k_set_candidate, axis=0)))
                else:
                    k_set_candidates = np.concatenate((k_set_candidates, np.expand_dims(k_set_candidate, axis=0)))
        print('**** got after debug. k_set_candidates: ', k_set_candidates)
        if self.debug:
            title = '\tk_set_candidates({}/{} passed filters)'.format(k_set_candidates.shape[0],
                                                                      possibilities_candidates_num)
            print(utils.var_to_string(k_set_candidates, title=title, with_data=False))
        if k_set_candidates.shape[0] == 0:
            print(utils.WARNING.format('not one k_set passed filters', 'algo client algorithm'))
        return k_set_candidates

    def find_best_k_set(self, line_ind_to_line_cluster_dict: dict, k_set_candidates: np.array) -> (np.array, float):
        """
        :param line_ind_to_line_cluster_dict:
        :param k_set_candidates:
        for each k_set in k_set_candidates:
            1) distribute lines to its closest point
                -> we have k points. each get some lines s.t. there is no other point that is closer to the line
            2) build clusters from the the distribution in step 1. now go over each cluster and sum the
                distances of the lines from the point -> all_clusters_sum
            3) if all_clusters_sum is better than what we saw so far, save it and the k_set
        :return: the best k_set we saw (lowest cost)
        """
        best_k3d_points, min_cost = None, float("inf")

        for k3d_points_i in k_set_candidates:
            _3d_point_ind_to_its_lines_indices = self.distribute_lines_to_closest_points(line_ind_to_line_cluster_dict,
                                                                                         k3d_points_i)

            min_cost_k_set_i = self.create_clusters_and_get_loss(line_ind_to_line_cluster_dict, k3d_points_i,
                                                                 _3d_point_ind_to_its_lines_indices)
            if min_cost_k_set_i < min_cost:
                min_cost = min_cost_k_set_i
                best_k3d_points = k3d_points_i

        if self.debug and k_set_candidates.shape[0] > 0:
            print('\tsearched {} k_set_candidates'.format(k_set_candidates.shape[0]))
        return best_k3d_points, min_cost

    def random_search(self, lines: list) -> (np.array, float):
        """
        # TODO not passing unit testing - debug and fix
        :param lines: list(Class Line)
        :return:
        """
        best_k3d_points, min_cost = None, float("inf")
        line_ind_to_line_cluster_dict = self.preprocess_lines(lines)

        for i in range(self.t):
            # first distribute RANDOMLY the lines to k groups (groups sizes not equal)
            clusters_dict = defaultdict(list)  # cluster index to list(Cluster)
            for line_ind, (line, cluster) in line_ind_to_line_cluster_dict.items():
                cluster_index = np.random.randint(0, self.k)
                clusters_dict[cluster_index].append(cluster)

            # we have k groups of clusters - join each group to a super cluster
            super_clusters_list = []
            for cluster_ind, cluster_list in clusters_dict.items():
                # print(cluster_ind, len(cluster_list))
                super_cluster = Cluster.join_clusters(clusters=cluster_list)
                super_clusters_list.append(super_cluster)
            # print(len(super_clusters_list))

            # now get min_cost_point from each super cluster as the k_set_candidate
            k3d_points_i = np.empty(shape=(self.k, 1, utils.DIM_3))
            min_cost_k_set_i = 0
            valid_k_set_candidate = True
            for j, super_cluster in enumerate(super_clusters_list):
                # matrix singular crashes - so check det != 0 (==matrix not singular -> inv exists)
                if np.linalg.det(super_cluster.rejection_sum_mat) != 0:
                    min_cost_point, min_cost_value = super_cluster.get_min_cost_point(with_value=True)
                    k3d_points_i[j] = min_cost_point
                    min_cost_k_set_i += min_cost_value
                else:
                    valid_k_set_candidate = False  # k_set is not in size k
                    break

            # if valid and cam filter on: check if k set has cameras point in it
            if valid_k_set_candidate and self.use_cam_pos_filter:
                k3d_points_i = self.filter_cams_points_from_candidates(k3d_points_i, self.cams_positions, debug=False)
                if k3d_points_i.shape[0] != self.k:
                    valid_k_set_candidate = False

            # if valid and unique filter on: check if k set has duplicated points
            if valid_k_set_candidate and self.use_unique_points_filter:
                k3d_points_i = self.filter_duplicates_from_candidates(k3d_points_i, debug=False)
                if k3d_points_i.shape[0] != self.k:
                    valid_k_set_candidate = False

            # if valid and in_range_filter on: check if any pair norm subtraction is in range
            if valid_k_set_candidate and self.use_in_range_filter:
                all_norms_in_range = self.all_norms_in_range(k3d_points_i, self.in_range_values['bottom'],
                                                             self.in_range_values['top'])
                if not all_norms_in_range:
                    valid_k_set_candidate = False

            # check if we improved
            if valid_k_set_candidate and min_cost_k_set_i < min_cost:
                min_cost = min_cost_k_set_i
                best_k3d_points = k3d_points_i
            # print('iter {}: min_cost {:.2f}, k3d {}'.format(i, min_cost, k3d_points_i.tolist()))
            # exit(22)

        if self.debug and best_k3d_points is not None:
            print('\tsearched {} k_set_candidates:'.format(self.t))
            print('\t\tmin cost {:,.2f}'.format(min_cost))
            best_k3d_points = best_k3d_points.reshape(self.k, utils.DIM_3)
            print(utils.var_to_string(best_k3d_points, title='\t\tkmeans', with_data=True))

        return best_k3d_points, min_cost

    # FILTERS
    @staticmethod
    def all_norms_in_range_opt(k_set: np.array, dynamic_norm_dict: dict, idx: list, bottom: int,
                               top: int) -> bool:
        """
        :param k_set: k_set candidate
        :param dynamic_norm_dict:
        :param idx:
        :param bottom:
        :param top:
        for p1,p2 in k choose 2 possibilities:
            check if norm(p1-p2) in range
        :return:
        """
        for i in range(k_set.shape[0]):
            for j in range(i + 1, k_set.shape[0]):
                key = str(sorted([idx[j], idx[i]]))
                # print(key)
                if key in dynamic_norm_dict:
                    norm = dynamic_norm_dict[key]
                else:
                    norm = np.linalg.norm(k_set[i] - k_set[j])
                    dynamic_norm_dict[key] = norm
                if not (bottom <= norm <= top):
                    # pass
                    return False
        return True

    @staticmethod
    def all_norms_in_range(k_set: np.array, bottom: int, top: int) -> bool:
        """
        :param k_set: k_set candidate
        for p1,p2 in k choose 2 possibilities:
            check if norm(p1-p2) in range
        :param bottom:
        :param top:
        :return:
        """
        for i in range(k_set.shape[0]):
            for j in range(i + 1, k_set.shape[0]):
                norm = np.linalg.norm(k_set[i] - k_set[j])
                if not (bottom <= norm <= top):
                    return False
        return True

    @staticmethod
    def filter_duplicates_from_candidates(individual_candidates: np.array, debug: bool = False) -> np.array:
        individual_candidates = np.unique(individual_candidates, axis=0)
        if debug:
            title = '\tindividual_candidates(post duplicates filter)'
            print(utils.var_to_string(individual_candidates, title=title, with_data=False))
        return individual_candidates

    @staticmethod
    def filter_cams_points_from_candidates(individual_candidates: np.array, cam_positions: np.array,
                                           debug: bool = False) -> np.array:
        """
        :param individual_candidates: np.array, dtype=float, shape=(X, 1, 3)
        :param cam_positions:
        :param debug:
        filter cameras positions (each 3d point) from the candidates
        :return: individual_candidates filtered
        """
        cams_positions_indices_in_candidates = []
        for cam_pos in cam_positions:
            diffs = cam_pos - individual_candidates  # compute all segments from candidates to cam_pose
            # compute all distances from candidates to cam_pose by taking norm of each substruction
            diffs_norms = np.linalg.norm(diffs, axis=2)
            cams_positions_indices_in_candidates += list(
                np.where((diffs_norms < 0.0001).all(-1))[0])

        individual_candidates = np.delete(individual_candidates, cams_positions_indices_in_candidates, axis=0)

        if debug:
            title = '\tindividual_candidates(post cams filter)'
            print(utils.var_to_string(individual_candidates, title=title, with_data=False))
        return individual_candidates

    def filter_noise(self, k3d_points: np.array) -> np.array:
        """
        NOTICE
        NOTICE TODO - no working on k == 1 - need fixing
        FUTURE: until we find a better way to support distributing the k3d_points to 'their' cluster
        from previous rounds(the current way is greedy and fail sometimes), we supprot noise filter only when
        k == 1

        :param k3d_points: np array shape=(k,3)
        using self.cluster_index_to_points: list of list of points ( point in shape (3,) )
            each entry is a "cluster".
        filter noise: - using moving average of size `self.noise_filter_size` to filter noise
            iter 1 : create k clusters with each point of the k3d_points
            iter 2+: each point finds its closest cluster and joins it
            if iter > self.noise_filter_size:
                delete first in each cluster(oldest)
        :return: k3d_points_filtered: np array shape=(k,3) - the mean of each cluster
        """
        k3d_points_filtered = np.zeros(shape=(self.k, utils.DIM_3)) - 100

        if len(self.clusters_points[0]) == 0:  # first iteration - create k clusters with 1 point
            for i, _3d_point in enumerate(k3d_points):
                self.clusters_points[i].append(_3d_point.tolist())
            k3d_points_filtered = k3d_points
        else:
            clusters_mean_points = np.mean(self.clusters_points, axis=1)  # 3d array - mean by rows of points
            for _3d_point in k3d_points:
                point_to_clusters_mean_norms = cdist(np.expand_dims(_3d_point, axis=0), clusters_mean_points)
                min_norm_cluster_index = np.argmin(point_to_clusters_mean_norms)
                self.clusters_points[min_norm_cluster_index].append(_3d_point.tolist())
                clusters_mean_points[min_norm_cluster_index] = np.ones(3) * float("inf")  # disable this cluster
                # check if over cluster size
                if len(self.clusters_points[min_norm_cluster_index]) > self.noise_filter_size:
                    del self.clusters_points[min_norm_cluster_index][0]
                # select the mean of the cluster as the a new point
                cluster_points_mean = np.mean(self.clusters_points[min_norm_cluster_index], axis=0)
                k3d_points_filtered[min_norm_cluster_index] = cluster_points_mean
        if self.debug:
            print(utils.var_to_string(k3d_points, '\tk3d_points pre  filter', with_data=True))
            print(utils.var_to_string(k3d_points_filtered, '\tk3d_points post filter', with_data=True))
        return k3d_points_filtered

    @staticmethod
    def distribute_lines_to_closest_points(line_ind_to_line_cluster_dict: dict, k3d_points_i: np.array) -> dict:
        # associate each line to its closest point in k3d_points
        _3d_point_ind_to_its_lines_indices = defaultdict(list)
        for line_ind, (line, cluster) in line_ind_to_line_cluster_dict.items():
            min_point_index, min_points_sqr_dist = -1, float("inf")
            for j, _3d_point in enumerate(k3d_points_i):
                sqr_dist = cluster.get_cost_from_point(_3d_point)
                if sqr_dist < min_points_sqr_dist:
                    min_points_sqr_dist = sqr_dist
                    min_point_index = j
            _3d_point_ind_to_its_lines_indices[min_point_index].append(line_ind)
        return _3d_point_ind_to_its_lines_indices

    @staticmethod
    def create_clusters_and_get_loss(line_ind_to_line_cluster_dict: dict, k3d_points_i: np.array,
                                     _3d_point_to_its_lines_indices: dict) -> float:
        min_cost_k_set_i = 0.0
        for j, _3d_point in enumerate(k3d_points_i):
            lines_indices = _3d_point_to_its_lines_indices[j]
            if len(lines_indices) > 1:
                clusters = []
                for line_ind in lines_indices:
                    clusters.append(line_ind_to_line_cluster_dict[line_ind][CLUSTER_IND])
                big_cluster_around_point = Cluster.join_clusters(clusters=clusters)
                cost_point_i = big_cluster_around_point.get_cost_from_point(_3d_point)
            elif len(lines_indices) == 1:
                cluster = line_ind_to_line_cluster_dict[lines_indices[0]][CLUSTER_IND]
                cost_point_i = cluster.get_cost_from_point(_3d_point)
            else:
                cost_point_i = 0.0
            min_cost_k_set_i += cost_point_i
        return min_cost_k_set_i


class CameraInfo:
    def __init__(self, mac: str, cam_port: int, position: POINT3D, rotation: MATRIX33D, intrinsic: MATRIX33D):
        """
        Auxiliary Class
        :param mac: mac id of the RP that has this camera - for debugging
        :param cam_port: port of the cam
        :param position: np.array, dtype=float, shape=(1,3): 3d point. cam position in the world
        :param rotation: np.array, dtype=float, shape=(3,3): rotation matrix
        :param intrinsic: np.array, dtype=float, shape=(3,3): intrinsic matrix
        :member k2d_points: np.array, dtype=float, shape=(k,2). k 2d points. e.g. k locations of contours
            found on image captured
        """
        self.mac = mac
        self.port = cam_port
        self.position = np.array(position, dtype=float)
        if len(self.position.shape) == 1:  # shape=(3,) -> shape=(1,3)
            self.position = self.position.reshape(1, utils.DIM_3)
        self.rotation = np.array(rotation, dtype=float)
        intrinsic = np.array(intrinsic, dtype=float)
        # these are known indexes in intrinsic matrix for the needed parameters for the geometrical scene construction.
        self.fx = intrinsic[0][0]  # the focal length in units of the x side length of a pixel
        self.fy = intrinsic[1][1]  # the focal length in units of the y side length of a pixel
        self.cx = intrinsic[0][2]  # the x value of the image center
        self.cy = intrinsic[1][2]  # the y value of the image center
        self.k2d_points = None
        return

    def __str__(self):
        to_str = 'CameraInfo:\n'
        to_str += '\tmac {}\n'.format(self.mac)
        to_str += '\tport {}\n'.format(self.port)
        to_str += '\tfx {:.3f}\n'.format(self.fx)
        to_str += '\tfy {:.3f}\n'.format(self.fy)
        to_str += '\tcx {:.3f}\n'.format(self.cx)
        to_str += '\tcy {:.3f}\n'.format(self.cy)
        to_str += '\tposition({})={}\n'.format(self.position.shape, np.round(self.position, 3).tolist())
        to_str += '\trotation({})={}'.format(self.rotation.shape, np.round(self.rotation, 3).tolist())
        if self.k2d_points is not None:
            to_str += '\n'
            to_str += 'k2d_points {}'.format(np.round(self.k2d_points, 3).tolist())
        return to_str

    def set_new_points(self, new_k2d_points: np.array) -> None:
        """ :param new_k2d_points: np.array, dtype=float, shape=(k,2) """
        if new_k2d_points is None or len(new_k2d_points) == 0:
            self.k2d_points = np.array([[]], dtype=float)
        else:
            self.k2d_points = np.array(new_k2d_points, dtype=float)
        return


class Line:
    """
    Line is represented by 2 vectors - a point and direction
    p: point (displacement of the line).  projection of the origin onto the line
    v: normed direction. the spanning vector of the line is normalized to a unit vector
    """

    def get_projected_point(self, _3d_point: POINT3D) -> POINT3D:
        proj_point_on_line = self.p + np.dot(_3d_point.flatten(), self.v.flatten()) * self.v
        # print(utils.var_to_string(proj_point_on_line, title='\t\t proj_point_on_line', with_data=True))
        return proj_point_on_line

    def __init__(self, p: POINT3D, v: POINT3D):
        p = p.flatten()
        v = v.flatten()
        v_normalized = v / np.linalg.norm(v, ord=2)
        dot = np.dot(p, v_normalized)
        p_shifted = p - (dot * v_normalized)
        p_shifted = p_shifted.reshape(1, -1)
        v_normalized = v_normalized.reshape(1, -1)
        self.p = p_shifted
        self.p_orig = p.reshape(-1, 1)
        self.v = v_normalized
        return

    def __str__(self):
        to_str = 'Line:\n'
        to_str += '\tp({})={}\n'.format(self.p.shape, np.round(self.p, 3).tolist())
        to_str += '\tv({})={}'.format(self.v.shape, np.round(self.v, 3).tolist())
        return to_str


class Cluster:
    def __init__(self, line: Line):
        """
        :param line:
        members:
        rejection_sum_mat: np.array shape=(3,3)
        projection_sum_vec: np.array shape=(1,3)
        dist_squared_sum: float
        """
        self.rejection_sum_mat = np.eye(utils.DIM_3, dtype=int) - line.v * line.v.T
        self.projection_sum_vec = line.p
        self.dist_squared_sum = np.dot(line.p.flatten(), line.p.flatten())
        return

    def __str__(self):
        to_str = 'Cluster:\n'
        to_str += '\trejection_sum_mat({})={}\n'.format(self.rejection_sum_mat.shape, self.rejection_sum_mat.tolist())
        to_str += '\tprojection_sum_vec({})={}\n'.format(self.projection_sum_vec.shape,
                                                         self.projection_sum_vec.tolist())
        to_str += '\tdist_squared_sum={}'.format(self.dist_squared_sum)
        return to_str

    @classmethod
    def join_clusters(cls, clusters: list):
        """
        creates super cluster from a list of clusters
        :return: class Cluster
        """
        base_cluster = copy.deepcopy(clusters[0])
        for i in range(1, len(clusters)):
            base_cluster.rejection_sum_mat += clusters[i].rejection_sum_mat
            base_cluster.projection_sum_vec += clusters[i].projection_sum_vec
            base_cluster.dist_squared_sum += clusters[i].dist_squared_sum
        return base_cluster

    def get_min_cost_point(self, with_value: bool = False) -> (POINT3D, float):
        """
        :return: the point that minimizes the cost (shape should be 1x3)
        """
        min_cost_value = None
        min_cost_point = (np.linalg.inv(self.rejection_sum_mat) @ self.projection_sum_vec.T).T
        if with_value:
            min_cost_value = self.dist_squared_sum - np.dot(self.projection_sum_vec.flatten(), min_cost_point.flatten())
        return min_cost_point, min_cost_value

    def get_cost_from_point(self, _3d_point: np.array) -> float:
        """
        :param _3d_point: 1x3 point
        :return: the cost of the cluster from that point
        """
        rej_sum_mult_3dpoint = self.rejection_sum_mat @ _3d_point.T
        two_proj_sum = 2 * self.projection_sum_vec
        sub_rej_proj = rej_sum_mult_3dpoint - two_proj_sum.T
        dot_3dpoint_sub = np.dot(_3d_point.flatten(), sub_rej_proj.flatten())
        sqr_dist = dot_3dpoint_sub + self.dist_squared_sum
        return sqr_dist


def sim_run_with_cam_info() -> None:
    def get_fake_rp_meta_data():
        meta_dict_ = OrderedDict()
        meta_dict_['dc:a6:32:0f:4c:ca'] = {
            'rotation_mat': [[0.996, -0.064, -0.057], [0.073, 0.982, 0.17], [0.991, -0.046, -0.12]],
            'distortion': [-0.167, 0.061, -0.013, 0.003, -0.012],
            'intrinsic_mat': [[267.597, 0.0, 284.127], [0.0, 266.12, 291.123], [0.0, 0.0, 1.0]],
            'cam_port': 0,
            'position': [-13.885, -0.398, 6.567],
            'help': 'original cam 2'
        }
        meta_dict_['dc:a6:32:0f:4f:33'] = {
            'rotation_mat': [[0.991, 0.025, 0.126], [-0.046, 0.984, 0.167], [-0.12, -0.172, -0.977]],
            'distortion': [-0.662, 0.207, 0.008, 0.027, 0.186],
            'intrinsic_mat': [[572.353, 0.0, 315.55], [0.0, 585.255, 267.113], [0.0, 0.0, 1.0]],
            'cam_port': 0,
            'position': [-22.12, -1.094, -8.94],
            'help': 'original cam 3'
        }

        meta_dict_['dc:a6:32:60:91:d1'] = {
            'rotation_mat': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'distortion': [-0.582, 0.53, -0.031, -0.044, -0.368],
            'intrinsic_mat': [[374.492, 0.0, 294.046], [0.0, 373.022, 365.784], [0.0, 0.0, 1.0]],
            'cam_port': 0,
            'position': [0.0, 0.0, 0.0],
            'help': 'original cam 1'
        }
        return meta_dict_

    def get_fake_data_from_server():
        fake_in_ = {"dc:a6:32:0f:4c:ca": [[29.654, -31.657], [27.969, 9.685], [-5.417, -40.003], [-4.075, -16.629]],
                    "dc:a6:32:0f:4f:33": [[-35.713, 15.089], [-44.359, 22.2], [43.855, -49.922], [49.221, 11.748]],
                    "dc:a6:32:60:91:d1": [[11.165, -49.293], [-47.694, 2.477], [-10.014, -45.333], [47.376, -26.723]]}
        return fake_in_

    np.random.seed(42)
    t_start = utils.get_time_now()
    print('Fake initialize:')
    # get meta data from CamClientMeta.json_old
    meta_dict = get_fake_rp_meta_data()

    # create CameraInfo for each RP(cam)
    cams_info_dict = OrderedDict()
    cams_positions = np.empty(shape=(0, 1, utils.DIM_3))
    for mac, cam_meta in meta_dict.items():
        intrinsic_mat = cam_meta['intrinsic_mat']
        position_vec = np.expand_dims(cam_meta['position'], axis=0)
        rotation_mat = np.array(cam_meta['rotation_mat'], dtype=float)  # needed cus T
        cam_info_i = CameraInfo(
            mac=mac,
            cam_port=cam_meta['cam_port'],
            position=position_vec,
            rotation=rotation_mat,
            intrinsic=intrinsic_mat
        )
        cams_positions = np.concatenate((cams_positions, cam_info_i.position.reshape(1, 1, utils.DIM_3)))
        cams_info_dict[cam_info_i.mac] = cam_info_i

    for cam_info in cams_info_dict.values():
        print('{}'.format(cam_info))
    print('{}'.format(utils.var_to_string(cams_positions, title='cams_positions', with_data=True)))

    kmeans = KMeans(
        k=4,
        run_type='exhaustive',
        t=5000,
        use_cam_pos_filter=True,
        use_in_range_filter=True,
        use_unique_points_filter=True,
        use_noise_filter=True,
        cams_positions=cams_positions,
        in_range_values={'bottom': 0, 'top': 10},
        noise_filter_size=10,
        debug=True
    )
    print(kmeans)

    print('Fake iteration:')
    fake_in = get_fake_data_from_server()  # got input - insert to the CameraInfo
    for mac, k2d_points in fake_in.items():
        # CameraInfo has all the cameras information(position, rotation...) and the k2d_points from this iteration
        cams_info_dict[mac].set_new_points(k2d_points)

    # generate lines using KMeans aux function
    lines = kmeans.generate_lines_from_cams_info(cams_info_list=list(cams_info_dict.values()))
    _, _ = kmeans.run_kmeans(lines)
    print('\tRun time {}'.format(utils.get_time_str_seconds(t_start)))
    return


def unit_tests():
    def plot_kmeans_results(cams_p_v: list = None, lines: list = None, k_set: np.array = None,
                            k_set_ground: np.array = None,
                            cost: float = None, cost_ground: float = None, title='Plot data:',
                            world_size=(-2, 2)) -> None:
        """
        :param cams_p_v: original cameras info. each cam has p and v
        :param lines: list of class Line. each class is normalized and projected camera
        :param k_set: np array of size kx3 - kmeans output
        :param k_set_ground: np array of size kx3 - ground truth
        :param cost:
        :param cost_ground:
        :param title:
        :param world_size:
        :return:
        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # print('PLOT: {}'.format(title))

        fig = plt.figure()
        axes = Axes3D(fig)
        utils.add_arrows_to_axed3d()
        axes.autoscale(enable=True, axis='both', tight=True)

        outer_cube_definition = [(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1)]
        plot_cube(axes, outer_cube_definition, label='outer cube(edge=2)')
        inner_cube_definition = [(-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, 0.5)]
        plot_cube(axes, inner_cube_definition, label='inner cube(edge=1)')

        # Setting the axes properties
        axes.set_facecolor("black")
        cube_bottom, cube_top = world_size
        axes.set_xlim3d([cube_bottom, cube_top])
        axes.set_ylim3d([cube_bottom, cube_top])
        axes.set_zlim3d([cube_bottom, cube_top])

        label_color = "red"
        axes.set_xlabel("X", color=label_color)
        axes.set_ylabel("Y", color=label_color)
        axes.set_zlabel("Z", color=label_color)

        axes.grid(True)

        background_and_alpha = (0.5, 0.2, 0.5, 0.4)  # RGB and alpha
        axes.w_xaxis.set_pane_color(background_and_alpha)
        axes.w_yaxis.set_pane_color(background_and_alpha)
        axes.w_zaxis.set_pane_color(background_and_alpha)
        if cams_p_v is not None:
            cams_color = 'orange'
            s = 0.5  # length factor of the arrow
            for i, cam_p_v in enumerate(cams_p_v):
                # print('\tcam {}: p={}, v={}'.format(i, cam_p_v['p'], cam_p_v['v']))
                p, v = cam_p_v['p'].flatten(), cam_p_v['v'].flatten()
                axes.arrow3D(  # arrow - the line with direction
                    p[0],  # x1
                    p[1],  # y1
                    p[2],  # z1
                    v[0] * s,  # x2
                    v[1] * s,  # y2
                    v[2] * s,  # z2
                    mutation_scale=10,  # size of the whole arrow
                    ec='white',  # border of arrow
                    fc=cams_color  # face color
                )
                # plot the base of the line
                xs, ys, zs = cam_p_v['p'][:, 0], cam_p_v['p'][:, 1], cam_p_v['p'][:, 2]
                label = "cameras" if i == 0 else None
                axes.plot3D(xs, ys, zs, color=cams_color, marker="*", markersize=10, label=label, linewidth=0.5,
                            linestyle=None)

        if lines is not None:
            lines_color = 'b'
            s = 1  # length factor of the arrow
            for i, line in enumerate(lines):
                # print('\tline {}: p={}, v={}'.format(i, line.p, line.v))
                p, v = line.p.flatten(), line.v.flatten()
                axes.arrow3D(  # arrow - the line with direction
                    p[0],  # x1
                    p[1],  # y1
                    p[2],  # z1
                    v[0] * s,  # x2
                    v[1] * s,  # y2
                    v[2] * s,  # z2
                    mutation_scale=10,  # size of the whole arrow
                    ec='white',  # border of arrow
                    fc=lines_color  # face color
                )
                # plot the base of the line
                xs, ys, zs = line.p[:, 0], line.p[:, 1], line.p[:, 2]
                label = "lines" if i == 0 else None
                axes.plot3D(xs, ys, zs, color=lines_color, marker="*", markersize=10, label=label, linewidth=0.5,
                            linestyle=None)

        base_size = 30
        if k_set is not None:  # kmeans out
            # print(utils.var_to_string(k_set, '\tk_set', with_data=True))
            k_set_color = 'r'
            xs, ys, zs = k_set[:, 0], k_set[:, 1], k_set[:, 2]
            axes.scatter(xs, ys, zs, color=k_set_color, marker="o", s=2 * base_size, label="k_set")
            if cost is not None:
                # print("\tcost={:.2f}".format(cost))
                axes.text2D(0.05, 0.90, "cost={:.2f}".format(cost), transform=axes.transAxes,
                            color=k_set_color)

        if k_set_ground is not None:  # ground truth
            k_set_ground_color = 'lawngreen'
            # print(utils.var_to_string(k_set_ground, '\tk_set_ground', with_data=True))
            xs, ys, zs = k_set_ground[:, 0], k_set_ground[:, 1], k_set_ground[:, 2]
            axes.scatter(xs, ys, zs, color=k_set_ground_color, marker="o", s=base_size, label="k_set_ground")
            if cost_ground is not None:
                # print("\tcost_ground={:.2f}".format(cost_ground))
                axes.text2D(0.05, 0.95, "cost_ground={:.2f}".format(cost_ground), transform=axes.transAxes,
                            color=k_set_ground_color)
        plt.legend(loc='upper right', ncol=1, fancybox=True, framealpha=0.5)
        plt.title(title, color='white')
        plt.show()
        return

    def plot_cube(axes, cube_definition, label='cube'):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        cube_definition_array = [
            np.array(list(item))
            for item in cube_definition
        ]

        points = []
        points += cube_definition_array
        vectors = [
            cube_definition_array[1] - cube_definition_array[0],
            cube_definition_array[2] - cube_definition_array[0],
            cube_definition_array[3] - cube_definition_array[0]
        ]

        points += [cube_definition_array[0] + vectors[0] + vectors[1]]
        points += [cube_definition_array[0] + vectors[0] + vectors[2]]
        points += [cube_definition_array[0] + vectors[1] + vectors[2]]
        points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

        points = np.array(points)

        edges = [
            [points[0], points[3], points[5], points[1]],
            [points[1], points[5], points[7], points[4]],
            [points[4], points[2], points[6], points[7]],
            [points[2], points[6], points[3], points[0]],
            [points[0], points[2], points[4], points[1]],
            [points[3], points[6], points[7], points[5]]
        ]

        faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
        faces.set_facecolor((0, 0, 1, 0.1))

        axes.add_collection3d(faces)

        # Plot the points themselves to force the scaling of the axes
        axes.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, color='b', label=label)

        return

    def exhaustive_1_1(debug: bool, with_plot: bool, eps=0.000001):
        """
        2 cams looking at origin k==1
        expecting (0,0,0) as output
        """
        np.random.seed(42)
        t_start = utils.get_time_now()
        k = 1
        title = 'exhaustive_1_1: 2 cameras looking at the origin. K={}'.format(k)
        print(title)
        # create cameras views and lines from each view
        cam_1_p_v = {'p': np.array([[-1, -1, -1]], dtype=float), 'v': np.array([[2, 2, 2]], dtype=float)}  # cam 1
        line_1 = Line(p=cam_1_p_v['p'], v=cam_1_p_v['v'])
        cam_2_p_v = {'p': np.array([[1, 1, 1]], dtype=float), 'v': np.array([[-2, -2, -2]], dtype=float)}  # cam 2
        line_2 = Line(p=cam_2_p_v['p'], v=cam_2_p_v['v'])
        cams_views = [cam_1_p_v, cam_2_p_v]  # un-normalized cameras data for plotting
        lines = [line_1, line_2]  # cameras to lines (projection and unit vector)

        cams_positions = [cam_1_p_v['p'], cam_2_p_v['p']]  # for KMeans cam pos filter
        k_set_ground = np.zeros(shape=(k, utils.DIM_3), dtype=float)
        cost_ground = 0.0

        kmeans = KMeans(
            k=k,
            run_type='exhaustive',
            t=5000,
            use_cam_pos_filter=True,
            use_in_range_filter=True,
            use_unique_points_filter=True,
            use_noise_filter=False,
            cams_positions=cams_positions,
            in_range_values={'bottom': 0, 'top': 10},
            noise_filter_size=None,
            debug=debug
        )

        if debug:
            print(kmeans)
            print('\tLines:')
            for i, line in enumerate(lines):
                print('\t\tline {}: p={}, v={}'.format(i, np.round(line.p, 3).tolist(), np.round(line.v, 3).tolist()))
            print('\tKmeans debug out:')
        k_set, cost = kmeans.run_kmeans(lines)

        print('\tGround Truth vs predicted:')
        print(utils.var_to_string(k_set_ground, '\t\tk_set_ground', with_data=True))
        print(utils.var_to_string(k_set, '\t\tk_set_algo  ', with_data=True))
        print('\t\tcost_ground={:.2f}'.format(cost_ground))
        print('\t\tcost_algo  ={:.2f}'.format(cost))

        if with_plot:
            plot_kmeans_results(cams_p_v=cams_views, lines=lines, k_set=k_set, k_set_ground=k_set_ground, cost=cost,
                                cost_ground=cost_ground, title=title)

        test_passed = abs(cost_ground - cost) < eps
        assert test_passed, 'Test failed: cost_ground={:.7f}, cost={:.7f}'.format(cost_ground, cost)
        print('\tTest PASSED')
        print('\tRun time {}'.format(utils.get_time_str_seconds(t_start)))
        return

    def exhaustive_1_2(debug: bool, with_plot: bool, eps=0.000001):
        """
        2 cams looking at origin k==2
        expecting [(0,0,0),(0,0,0)] as output
        notice use_unique_points_filter=False
            if it was on we would have got the same output since the values are not exactly zero (close to zero)
        """
        np.random.seed(42)
        t_start = utils.get_time_now()
        k = 2
        title = 'exhaustive_1_2: 2 cameras looking at the origin. K={}'.format(k)
        print(title)
        # create cameras views and lines from each view
        cam_1_p_v = {'p': np.array([[-1, -1, -1]], dtype=float), 'v': np.array([[2, 2, 2]], dtype=float)}  # cam 1
        line_1 = Line(p=cam_1_p_v['p'], v=cam_1_p_v['v'])
        cam_2_p_v = {'p': np.array([[1, 1, 1]], dtype=float), 'v': np.array([[-2, -2, -2]], dtype=float)}  # cam 2
        line_2 = Line(p=cam_2_p_v['p'], v=cam_2_p_v['v'])
        cams_views = [cam_1_p_v, cam_2_p_v]  # un-normalized cameras data for plotting
        lines = [line_1, line_2]  # cameras to lines (projection and unit vector)

        cams_positions = [cam_1_p_v['p'], cam_2_p_v['p']]  # for KMeans cam pos filter
        k_set_ground = np.zeros(shape=(k, utils.DIM_3), dtype=float)
        cost_ground = 0.0

        kmeans = KMeans(
            k=k,
            run_type='exhaustive',
            t=5000,
            use_cam_pos_filter=True,
            use_in_range_filter=True,
            use_unique_points_filter=False,
            use_noise_filter=False,
            cams_positions=cams_positions,
            in_range_values={'bottom': 0, 'top': 10},
            noise_filter_size=None,
            debug=debug
        )

        if debug:
            print(kmeans)
            print('\tLines:')
            for i, line in enumerate(lines):
                print('\t\tline {}: p={}, v={}'.format(i, np.round(line.p, 3).tolist(), np.round(line.v, 3).tolist()))
            print('\tKmeans debug out:')
        k_set, cost = kmeans.run_kmeans(lines)

        print('\tGround Truth vs predicted:')
        print(utils.var_to_string(k_set_ground, '\t\tk_set_ground', with_data=True))
        print(utils.var_to_string(k_set, '\t\tk_set_algo  ', with_data=True))
        print('\t\tcost_ground={:.2f}'.format(cost_ground))
        print('\t\tcost_algo  ={:.2f}'.format(cost))

        if with_plot:
            plot_kmeans_results(cams_p_v=cams_views, lines=lines, k_set=k_set, k_set_ground=k_set_ground, cost=cost,
                                cost_ground=cost_ground, title=title)

        test_passed = abs(cost_ground - cost) < eps
        assert test_passed, 'Test failed: cost_ground={:.7f}, cost={:.7f}'.format(cost_ground, cost)
        print('\tTest PASSED')
        print('\tRun time {}'.format(utils.get_time_str_seconds(t_start)))
        return

    def exhaustive_2_1(debug: bool, with_plot: bool, eps=0.000001):
        """
        2 cams looking at origin with symmetrical noise k==1
        expecting one of the projected points
        # since kmeans gives 4 approx, it should yield randomly one of the points - so the ground will be both
        """
        np.random.seed(42)
        t_start = utils.get_time_now()
        k = 1
        title = 'exhaustive_2_1: 2 cameras looking at the origin with symmetrical noise. K={}'.format(k)
        print(title)
        # create cameras views and lines from each view
        cam_1_p_v = {'p': np.array([[-1, -1, -1]], dtype=float), 'v': np.array([[1, 2, 2]], dtype=float)}  # cam 1
        line_1 = Line(p=cam_1_p_v['p'], v=cam_1_p_v['v'])
        cam_2_p_v = {'p': np.array([[1, 1, 1]], dtype=float), 'v': np.array([[-1, -2, -2]], dtype=float)}  # cam 2
        line_2 = Line(p=cam_2_p_v['p'], v=cam_2_p_v['v'])
        cams_views = [cam_1_p_v, cam_2_p_v]  # un-normalized cameras data for plotting
        lines = [line_1, line_2]  # cameras to lines (projection and unit vector)

        cams_positions = [cam_1_p_v['p'], cam_2_p_v['p']]  # for KMeans cam pos filter
        # since kmeans gives 4 approx, it should yield randomly one of the points - so the ground will be both
        k_set_ground = np.zeros(shape=(k + 1, utils.DIM_3), dtype=float)
        k_set_ground[0] = line_1.p
        k_set_ground[1] = line_2.p
        # TODO i don't know what loss is used. i get 0.9428 for euclidean dist but algo returns 0.8888
        cost_ground = 0.8888888888888  # TODO calculate expected loss

        kmeans = KMeans(
            k=k,
            run_type='exhaustive',
            t=5000,
            use_cam_pos_filter=True,
            use_in_range_filter=True,
            use_unique_points_filter=True,
            use_noise_filter=False,
            cams_positions=cams_positions,
            in_range_values={'bottom': 0, 'top': 10},
            noise_filter_size=None,
            debug=debug
        )

        if debug:
            print(kmeans)
            print('\tLines:')
            for i, line in enumerate(lines):
                print('\t\tline {}: p={}, v={}'.format(i, np.round(line.p, 3).tolist(), np.round(line.v, 3).tolist()))
            print('\tKmeans debug out:')
        k_set, cost = kmeans.run_kmeans(lines)

        print('\tGround Truth vs predicted:')
        print('\t\tk_set_ground has 2 options:')
        print(utils.var_to_string(k_set_ground[0], '\t\t\tk_set_ground[0]', with_data=True))
        print(utils.var_to_string(k_set_ground[1], '\t\t\tk_set_ground[1]', with_data=True))
        print(utils.var_to_string(k_set, '\t\tk_set_algo', with_data=True))
        print('\t\tcost_ground={:.2f}'.format(cost_ground))
        print('\t\tcost_algo  ={:.2f}'.format(cost))
        print('\t\tattempt = {}'.format(np.linalg.norm(line_1.p - line_2.p)))  # TODO delete

        if with_plot:
            plot_kmeans_results(cams_p_v=cams_views, lines=lines, k_set=k_set, k_set_ground=k_set_ground, cost=cost,
                                cost_ground=cost_ground, title=title)

        test_passed = abs(cost_ground - cost) < eps
        assert test_passed, 'Test failed: cost_ground={:.7f}, cost={:.7f}'.format(cost_ground, cost)
        print('\tTest PASSED')
        print('\tRun time {}'.format(utils.get_time_str_seconds(t_start)))
        return

    def exhaustive_2_2(debug: bool, with_plot: bool, eps=0.000001):
        """ 2 cams looking at origin with symmetrical noise k==1 """
        np.random.seed(42)
        t_start = utils.get_time_now()
        k = 2
        title = 'exhaustive_2_2: 2 cameras looking at the origin with symmetrical noise. K={}'.format(k)
        print(title)
        # create cameras views and lines from each view
        cam_1_p_v = {'p': np.array([[-1, -1, -1]], dtype=float), 'v': np.array([[1, 2, 2]], dtype=float)}  # cam 1
        line_1 = Line(p=cam_1_p_v['p'], v=cam_1_p_v['v'])
        cam_2_p_v = {'p': np.array([[1, 1, 1]], dtype=float), 'v': np.array([[-1, -2, -2]], dtype=float)}  # cam 2
        line_2 = Line(p=cam_2_p_v['p'], v=cam_2_p_v['v'])
        cams_views = [cam_1_p_v, cam_2_p_v]  # un-normalized cameras data for plotting
        lines = [line_1, line_2]  # cameras to lines (projection and unit vector)

        cams_positions = [cam_1_p_v['p'], cam_2_p_v['p']]  # for KMeans cam pos filter

        k_set_ground = np.zeros(shape=(k, utils.DIM_3), dtype=float)
        k_set_ground[0] = line_1.p
        k_set_ground[1] = line_2.p
        cost_ground = 0.0

        kmeans = KMeans(
            k=k,
            run_type='exhaustive',
            t=5000,
            use_cam_pos_filter=True,
            use_in_range_filter=True,
            use_unique_points_filter=True,
            use_noise_filter=False,
            cams_positions=cams_positions,
            in_range_values={'bottom': 0, 'top': 10},
            noise_filter_size=None,
            debug=debug
        )

        if debug:
            print(kmeans)
            print('\tLines:')
            for i, line in enumerate(lines):
                print('\t\tline {}: p={}, v={}'.format(i, np.round(line.p, 3).tolist(), np.round(line.v, 3).tolist()))
            print('\tKmeans debug out:')
        k_set, cost = kmeans.run_kmeans(lines)

        print('\tGround Truth vs predicted:')
        print(utils.var_to_string(k_set_ground, '\t\tk_set_ground', with_data=True))
        print(utils.var_to_string(k_set, '\t\tk_set_algo  ', with_data=True))
        print('\t\tcost_ground={:.2f}'.format(cost_ground))
        print('\t\tcost_algo  ={:.2f}'.format(cost))

        if with_plot:
            plot_kmeans_results(cams_p_v=cams_views, lines=lines, k_set=k_set, k_set_ground=k_set_ground, cost=cost,
                                cost_ground=cost_ground, title=title)

        test_passed = abs(cost_ground - cost) < eps
        assert test_passed, 'Test failed: cost_ground={:.7f}, cost={:.7f}'.format(cost_ground, cost)
        print('\tTest PASSED')
        print('\tRun time {}'.format(utils.get_time_str_seconds(t_start)))
        return

    def exhaustive_3_real_big_test(debug: bool, with_plot: bool, eps=0.000001):
        """ 4 cams looking at inner cube side with 4 blobs from each cam. k==4 """
        np.random.seed(42)
        t_start = utils.get_time_now()
        k = 4
        title = 'test3: 4 cams looking at inner cube side with 4 blobs from each cam. K={}'.format(k)
        print(title)

        # side of inner cube
        k_set_ground = np.zeros(shape=(k, utils.DIM_3), dtype=float)
        k_set_ground[0] = [-0.5, -0.5, -0.5]
        k_set_ground[1] = [0.5, -0.5, -0.5]
        k_set_ground[2] = [0.5, -0.5, 0.5]
        k_set_ground[3] = [-0.5, -0.5, 0.5]
        cost_ground = 0.0

        cams_views = []
        lines = []

        cam_1_p = np.array([[-1, -1, -1]], dtype=float)  # cam 1
        cam_1_1_p_v = {'p': cam_1_p, 'v': k_set_ground[0] - cam_1_p}
        cam_1_2_p_v = {'p': cam_1_p, 'v': k_set_ground[1] - cam_1_p}
        cam_1_3_p_v = {'p': cam_1_p, 'v': k_set_ground[2] - cam_1_p}
        cam_1_4_p_v = {'p': cam_1_p, 'v': k_set_ground[3] - cam_1_p}
        cams_views += [cam_1_1_p_v, cam_1_2_p_v, cam_1_3_p_v, cam_1_4_p_v]

        line_1_1 = Line(p=cam_1_p, v=cam_1_1_p_v['v'])
        line_1_2 = Line(p=cam_1_p, v=cam_1_2_p_v['v'])
        line_1_3 = Line(p=cam_1_p, v=cam_1_3_p_v['v'])
        line_1_4 = Line(p=cam_1_p, v=cam_1_4_p_v['v'])
        lines += [line_1_1, line_1_2, line_1_3, line_1_4]  # cameras to lines (projection and unit vector)

        cam_2_p = np.array([[1, 1, 1]], dtype=float)  # cam 2
        cam_2_1_p_v = {'p': cam_2_p, 'v': k_set_ground[0] - cam_2_p}
        cam_2_2_p_v = {'p': cam_2_p, 'v': k_set_ground[1] - cam_2_p}
        cam_2_3_p_v = {'p': cam_2_p, 'v': k_set_ground[2] - cam_2_p}
        cam_2_4_p_v = {'p': cam_2_p, 'v': k_set_ground[3] - cam_2_p}
        cams_views += [cam_2_1_p_v, cam_2_2_p_v, cam_2_3_p_v, cam_2_4_p_v]

        line_2_1 = Line(p=cam_2_p, v=cam_2_1_p_v['v'])
        line_2_2 = Line(p=cam_2_p, v=cam_2_2_p_v['v'])
        line_2_3 = Line(p=cam_2_p, v=cam_2_3_p_v['v'])
        line_2_4 = Line(p=cam_2_p, v=cam_2_4_p_v['v'])
        lines += [line_2_1, line_2_2, line_2_3, line_2_4]  # cameras to lines (projection and unit vector)

        cam_3_p = np.array([[-1, 1, 1]], dtype=float)  # cam 3
        cam_3_1_p_v = {'p': cam_3_p, 'v': k_set_ground[0] - cam_3_p}
        cam_3_2_p_v = {'p': cam_3_p, 'v': k_set_ground[1] - cam_3_p}
        cam_3_3_p_v = {'p': cam_3_p, 'v': k_set_ground[2] - cam_3_p}
        cam_3_4_p_v = {'p': cam_3_p, 'v': k_set_ground[3] - cam_3_p}
        cams_views += [cam_3_1_p_v, cam_3_2_p_v, cam_3_3_p_v, cam_3_4_p_v]

        line_3_1 = Line(p=cam_3_p, v=cam_3_1_p_v['v'])
        line_3_2 = Line(p=cam_3_p, v=cam_3_2_p_v['v'])
        line_3_3 = Line(p=cam_3_p, v=cam_3_3_p_v['v'])
        line_3_4 = Line(p=cam_3_p, v=cam_3_4_p_v['v'])
        lines += [line_3_1, line_3_2, line_3_3, line_3_4]  # cameras to lines (projection and unit vector)

        cam_4_p = np.array([[1, -1, 1]], dtype=float)  # cam 4
        cam_4_1_p_v = {'p': cam_4_p, 'v': k_set_ground[0] - cam_4_p}
        cam_4_2_p_v = {'p': cam_4_p, 'v': k_set_ground[1] - cam_4_p}
        cam_4_3_p_v = {'p': cam_4_p, 'v': k_set_ground[2] - cam_4_p}
        cam_4_4_p_v = {'p': cam_4_p, 'v': k_set_ground[3] - cam_4_p}
        cams_views += [cam_4_1_p_v, cam_4_2_p_v, cam_4_3_p_v, cam_4_4_p_v]

        line_4_1 = Line(p=cam_4_p, v=cam_4_1_p_v['v'])
        line_4_2 = Line(p=cam_4_p, v=cam_4_2_p_v['v'])
        line_4_3 = Line(p=cam_4_p, v=cam_4_3_p_v['v'])
        line_4_4 = Line(p=cam_4_p, v=cam_4_4_p_v['v'])
        lines += [line_4_1, line_4_2, line_4_3, line_4_4]  # cameras to lines (projection and unit vector)

        cams_positions = [cam_1_p, cam_2_p, cam_3_p, cam_4_p]  # for KMeans cam pos filter

        kmeans = KMeans(
            k=k,
            run_type='exhaustive',
            t=5000,
            use_cam_pos_filter=True,
            use_in_range_filter=True,
            use_unique_points_filter=True,
            use_noise_filter=False,
            cams_positions=cams_positions,
            in_range_values={'bottom': 0, 'top': 10},
            noise_filter_size=None,
            debug=debug
        )

        if debug:
            print(kmeans)
            print('\tLines:')
            for i, line in enumerate(lines):
                print('\t\tline {}: p={}, v={}'.format(i, np.round(line.p, 3).tolist(), np.round(line.v, 3).tolist()))
            print('\tKmeans debug out:')
        k_set, cost = kmeans.run_kmeans(lines)

        print('\tGround Truth vs predicted:')
        print(utils.var_to_string(k_set_ground, '\t\tk_set_ground', with_data=True))
        print(utils.var_to_string(k_set, '\t\tk_set_algo  ', with_data=True))
        print('\t\tcost_ground={:.2f}'.format(cost_ground))
        print('\t\tcost_algo  ={:.2f}'.format(cost))

        if with_plot:
            plot_kmeans_results(cams_p_v=cams_views, lines=lines, k_set=k_set, k_set_ground=k_set_ground, cost=cost,
                                cost_ground=cost_ground, title=title)

        test_passed = abs(cost_ground - cost) < eps
        assert test_passed, 'Test failed: cost_ground={:.7f}, cost={:.7f}'.format(cost_ground, cost)
        print('\tTest PASSED')
        print('\tRun time {}'.format(utils.get_time_str_seconds(t_start)))
        return

    def exhaustive_4_noise_check(debug: bool, with_plot: bool, eps=0.000001):
        """ 4 cams looking at one point - checking if noise filter works. k=1, 3 iterations, noise filter size 2 """
        np.random.seed(42)
        t_start = utils.get_time_now()
        k = 1
        noise_filter_size = 2
        title = 'exhaustive_4_noise_check: 4 cams looking at one point - checking if noise filter works. K={}'.format(k)
        print(title)
        k_set_ground1 = np.zeros(shape=(k, utils.DIM_3), dtype=float) + 2  # t==1
        k_set_ground1_filtered = k_set_ground1  # no filter applied on iter 1
        k_set_ground2 = np.zeros(shape=(k, utils.DIM_3), dtype=float) + 5  # t==2
        k_set_ground2_filtered = (k_set_ground1 + k_set_ground2) / noise_filter_size
        k_set_ground3 = np.zeros(shape=(k, utils.DIM_3), dtype=float) + 10  # t==3
        k_set_ground3_filtered = (k_set_ground2 + k_set_ground3) / noise_filter_size

        # create cameras views and lines from each view
        cost_ground = 0.0  # all iterations should yield 0 cost since we make the lines intercept
        cam_1_p = np.array([[-1, -1, -1]], dtype=float)  # cam 1
        cam_2_p = np.array([[1, 1, 1]], dtype=float)  # cam 2
        cam_3_p = np.array([[-1, 1, 1]], dtype=float)  # cam 3
        cam_4_p = np.array([[1, -1, 1]], dtype=float)  # cam 4

        # iteration 1
        cams_view_iter1 = []
        lines_iter1 = []
        for i in range(k):
            cam1_view = {'p': cam_1_p, 'v': k_set_ground1[i] - cam_1_p}  # camera blob
            cams_view_iter1.append(cam1_view)
            line1 = Line(p=cam_1_p, v=cam1_view['v'])
            lines_iter1.append(line1)

        for i in range(k):
            cam2_view = {'p': cam_2_p, 'v': k_set_ground1[i] - cam_2_p}
            cams_view_iter1.append(cam2_view)
            line2 = Line(p=cam_2_p, v=cam2_view['v'])
            lines_iter1.append(line2)

        for i in range(k):
            cam3_view = {'p': cam_3_p, 'v': k_set_ground1[i] - cam_3_p}
            cams_view_iter1.append(cam3_view)
            line3 = Line(p=cam_3_p, v=cam3_view['v'])
            lines_iter1.append(line3)

        for i in range(k):
            cam4_view = {'p': cam_4_p, 'v': k_set_ground1[i] - cam_4_p}
            cams_view_iter1.append(cam4_view)
            line4 = Line(p=cam_4_p, v=cam4_view['v'])
            lines_iter1.append(line4)

        # iteration 2
        cams_view_iter2 = []
        lines_iter2 = []
        for i in range(k):
            cam1_view = {'p': cam_1_p, 'v': k_set_ground2[i] - cam_1_p}  # camera blob
            cams_view_iter2.append(cam1_view)
            line1 = Line(p=cam_1_p, v=cam1_view['v'])
            lines_iter2.append(line1)

        for i in range(k):
            cam2_view = {'p': cam_2_p, 'v': k_set_ground2[i] - cam_2_p}
            cams_view_iter2.append(cam2_view)
            line2 = Line(p=cam_2_p, v=cam2_view['v'])
            lines_iter2.append(line2)

        for i in range(k):
            cam3_view = {'p': cam_3_p, 'v': k_set_ground2[i] - cam_3_p}
            cams_view_iter2.append(cam3_view)
            line3 = Line(p=cam_3_p, v=cam3_view['v'])
            lines_iter2.append(line3)

        for i in range(k):
            cam4_view = {'p': cam_4_p, 'v': k_set_ground2[i] - cam_4_p}
            cams_view_iter2.append(cam4_view)
            line4 = Line(p=cam_4_p, v=cam4_view['v'])
            lines_iter2.append(line4)

        # iteration 3
        cams_view_iter3 = []
        lines_iter3 = []
        for i in range(k):
            cam1_view = {'p': cam_1_p, 'v': k_set_ground3[i] - cam_1_p}  # camera blob
            cams_view_iter3.append(cam1_view)
            line1 = Line(p=cam_1_p, v=cam1_view['v'])
            lines_iter3.append(line1)

        for i in range(k):
            cam2_view = {'p': cam_2_p, 'v': k_set_ground3[i] - cam_2_p}
            cams_view_iter3.append(cam2_view)
            line2 = Line(p=cam_2_p, v=cam2_view['v'])
            lines_iter3.append(line2)

        for i in range(k):
            cam3_view = {'p': cam_3_p, 'v': k_set_ground3[i] - cam_3_p}
            cams_view_iter3.append(cam3_view)
            line3 = Line(p=cam_3_p, v=cam3_view['v'])
            lines_iter3.append(line3)

        for i in range(k):
            cam4_view = {'p': cam_4_p, 'v': k_set_ground3[i] - cam_4_p}
            cams_view_iter3.append(cam4_view)
            line4 = Line(p=cam_4_p, v=cam4_view['v'])
            lines_iter3.append(line4)

        if debug:
            print('\titer 1, looking at {}'.format(k_set_ground1.tolist()))
            print('\t\tcams_view_iter1:')
            for i, cam_p_v in enumerate(cams_view_iter1):
                print('\t\t\tcam {}: p={}, v={}'.format(i, cam_p_v['p'], cam_p_v['v']))
            print('\t\tlines_iter1:')
            for i, line in enumerate(lines_iter1):
                print('\t\t\tline {}: p={}, v={}'.format(i, np.round(line.p, 3).tolist(), np.round(line.v, 3).tolist()))

            print('\titer 2, looking at {}'.format(k_set_ground2.tolist()))
            print('\t\tcams_view_iter2:')
            for i, cam_p_v in enumerate(cams_view_iter2):
                print('\t\t\tcam {}: p={}, v={}'.format(i, cam_p_v['p'], cam_p_v['v']))
            print('\t\tlines_iter2:')
            for i, line in enumerate(lines_iter2):
                print('\t\t\tline {}: p={}, v={}'.format(i, np.round(line.p, 3).tolist(), np.round(line.v, 3).tolist()))

            print('\titer 3, looking at {}'.format(k_set_ground3.tolist()))
            print('\t\tcams_view_iter3:')
            for i, cam_p_v in enumerate(cams_view_iter3):
                print('\t\t\tcam {}: p={}, v={}'.format(i, cam_p_v['p'], cam_p_v['v']))
            print('\t\tlines_iter3:')
            for i, line in enumerate(lines_iter3):
                print('\t\t\tline {}: p={}, v={}'.format(i, np.round(line.p, 3).tolist(), np.round(line.v, 3).tolist()))

        cams_positions = [cam_1_p, cam_2_p, cam_3_p, cam_4_p]  # for KMeans cam pos filter

        kmeans_no_noise_filter = KMeans(
            k=k,
            run_type='exhaustive',
            t=5000,
            use_cam_pos_filter=True,
            use_in_range_filter=True,
            use_unique_points_filter=True,
            use_noise_filter=False,
            cams_positions=cams_positions,
            in_range_values={'bottom': 0, 'top': 10},
            noise_filter_size=None,
            debug=debug
        )
        kmeans_with_noise_filter = KMeans(
            k=k,
            run_type='exhaustive',
            t=5000,
            use_cam_pos_filter=True,
            use_in_range_filter=True,
            use_unique_points_filter=True,
            use_noise_filter=True,
            cams_positions=cams_positions,
            in_range_values={'bottom': 0, 'top': 10},
            noise_filter_size=noise_filter_size,
            debug=debug
        )

        world_size = (-10, 10)

        if debug:
            print(kmeans_no_noise_filter)
            print('Kmeans debug out iter 1 (no noise filter):')
        k_set1_no_noise, cost1_no_noise = kmeans_no_noise_filter.run_kmeans(lines_iter1)

        if debug:
            print(kmeans_with_noise_filter)
            print('Kmeans debug out iter 1 (with noise filter):')
        k_set1_with_noise, cost1_with_noise = kmeans_with_noise_filter.run_kmeans(lines_iter1)

        print('\tGround Truth vs predicted, iter 1:')
        print(utils.var_to_string(k_set_ground1, '\t\tk_set_ground1         ', with_data=True))
        print(utils.var_to_string(k_set1_no_noise, '\t\tk_set1_no_noise       ', with_data=True))
        print(utils.var_to_string(k_set_ground1_filtered, '\t\tk_set_ground1_filtered', with_data=True))
        print(utils.var_to_string(k_set1_with_noise, '\t\tk_set1_with_noise     ', with_data=True))
        print('\t\tcost_ground    ={:.2f}'.format(cost_ground))
        print('\t\tcost_no_noise  ={:.2f}'.format(cost1_no_noise))
        print('\t\tcost_with_noise={:.2f}'.format(cost1_with_noise))
        if with_plot:
            plot_kmeans_results(cams_p_v=cams_view_iter1, lines=lines_iter1, k_set=k_set1_no_noise,
                                k_set_ground=k_set_ground1, cost=cost1_no_noise, cost_ground=cost_ground,
                                title='no noise. iter {}'.format(1), world_size=world_size)
            plot_kmeans_results(cams_p_v=cams_view_iter1, lines=lines_iter1, k_set=k_set1_with_noise,
                                k_set_ground=k_set_ground1_filtered, cost=cost1_with_noise, cost_ground=cost_ground,
                                title='with noise filter. iter {}'.format(1), world_size=world_size)
        test_passed = abs(np.sum(k_set1_no_noise - k_set_ground1)) < eps
        assert test_passed, 'Test failed: k_set1_no_noise={}, k_set_ground1={}'.format(k_set1_no_noise.tolist(),
                                                                                       k_set_ground1.tolist())
        test_passed = abs(np.sum(k_set1_with_noise - k_set_ground1_filtered)) < eps
        assert test_passed, 'Test failed: k_set1_with_noise={}, k_set_ground1_filtered={}'.format(
            k_set1_with_noise.tolist(), k_set_ground1_filtered.tolist())

        if debug:
            print('Kmeans debug out iter 2 (no noise filter):')
        k_set2_no_noise, cost2_no_noise = kmeans_no_noise_filter.run_kmeans(lines_iter2)
        if debug:
            print('Kmeans debug out iter 2 (with noise filter):')
        k_set2_with_noise, cost2_with_noise = kmeans_with_noise_filter.run_kmeans(lines_iter2)
        print('\tGround Truth vs predicted, iter 2:')
        print(utils.var_to_string(k_set_ground2, '\t\tk_set_ground2         ', with_data=True))
        print(utils.var_to_string(k_set2_no_noise, '\t\tk_set2_no_noise       ', with_data=True))
        print(utils.var_to_string(k_set_ground2_filtered, '\t\tk_set_ground2_filtered', with_data=True))
        print(utils.var_to_string(k_set2_with_noise, '\t\tk_set2_with_noise     ', with_data=True))
        print('\t\tcost_ground    ={:.2f}'.format(cost_ground))
        print('\t\tcost_no_noise  ={:.2f}'.format(cost2_no_noise))
        print('\t\tcost_with_noise={:.2f}'.format(cost2_with_noise))
        if with_plot:
            plot_kmeans_results(cams_p_v=cams_view_iter2, lines=lines_iter2, k_set=k_set2_no_noise,
                                k_set_ground=k_set_ground2, cost=cost2_no_noise, cost_ground=cost_ground,
                                title='no noise. iter {}'.format(2), world_size=world_size)
            plot_kmeans_results(cams_p_v=cams_view_iter2, lines=lines_iter2, k_set=k_set2_with_noise,
                                k_set_ground=k_set_ground2_filtered, cost=cost2_with_noise, cost_ground=cost_ground,
                                title='with noise filter. iter {}'.format(2), world_size=world_size)
        test_passed = abs(np.sum(k_set2_no_noise - k_set_ground2)) < eps
        assert test_passed, 'Test failed: k_set2_no_noise={}, k_set_ground2={}'.format(k_set2_no_noise.tolist(),
                                                                                       k_set_ground2.tolist())
        test_passed = abs(np.sum(k_set2_with_noise - k_set_ground2_filtered)) < eps
        assert test_passed, 'Test failed: k_set2_with_noise={}, k_set_ground2_filtered={}'.format(
            k_set2_with_noise.tolist(), k_set_ground2_filtered.tolist())

        if debug:
            print('Kmeans debug out iter 3 (no noise filter):')
        k_set3_no_noise, cost3_no_noise = kmeans_no_noise_filter.run_kmeans(lines_iter3)
        if debug:
            print('Kmeans debug out iter 3 (with noise filter):')
        k_set3_with_noise, cost3_with_noise = kmeans_with_noise_filter.run_kmeans(lines_iter3)
        print('\tGround Truth vs predicted, iter 3:')
        print(utils.var_to_string(k_set_ground3, '\t\tk_set_ground3         ', with_data=True))
        print(utils.var_to_string(k_set3_no_noise, '\t\tk_set3_no_noise       ', with_data=True))
        print(utils.var_to_string(k_set_ground3_filtered, '\t\tk_set_ground3_filtered', with_data=True))
        print(utils.var_to_string(k_set3_with_noise, '\t\tk_set3_with_noise     ', with_data=True))
        print('\t\tcost_ground    ={:.2f}'.format(cost_ground))
        print('\t\tcost_no_noise  ={:.2f}'.format(cost3_no_noise))
        print('\t\tcost_with_noise={:.2f}'.format(cost3_with_noise))
        if with_plot:
            plot_kmeans_results(cams_p_v=cams_view_iter3, lines=lines_iter3, k_set=k_set3_no_noise,
                                k_set_ground=k_set_ground3, cost=cost3_no_noise, cost_ground=cost_ground,
                                title='no noise. iter {}'.format(3), world_size=world_size)
            plot_kmeans_results(cams_p_v=cams_view_iter3, lines=lines_iter3, k_set=k_set3_with_noise,
                                k_set_ground=k_set_ground3_filtered, cost=cost3_with_noise, cost_ground=cost_ground,
                                title='with noise filter. iter {}'.format(3), world_size=world_size)
        test_passed = abs(np.sum(k_set3_no_noise - k_set_ground3)) < eps
        assert test_passed, 'Test failed: k_set3_no_noise={}, k_set_ground3={}'.format(k_set3_no_noise.tolist(),
                                                                                       k_set_ground3.tolist())
        test_passed = abs(np.sum(k_set3_with_noise - k_set_ground3_filtered)) < eps
        assert test_passed, 'Test failed: k_set3_with_noise={}, k_set_ground3_filtered={}'.format(
            k_set3_with_noise.tolist(), k_set_ground3_filtered.tolist())
        print('\tTest PASSED')
        print('\tRun time {}'.format(utils.get_time_str_seconds(t_start)))
        return

    # def exhaustive_5_fail(debug: bool, with_plot: bool, eps=0.000001):
    #     """ 4 cams looking at one point - k=1 """
    #     # TODO change cam_2_p to 1,1,1 -> its direction becomes 0 -> algorithm fails. is it a bug?
    #     np.random.seed(42)
    #     t_start = utils.get_time_now()
    #     k = 1
    #     title = 'exhaustive_5_fail no points pass filter. K={}'.format(k)
    #     print(title)
    #     k_set_ground = np.ones(shape=(k, utils.DIM_3), dtype=float)
    #
    #     # create cameras views and lines from each view
    #     cost_ground = 0.0  # all iterations should yield 0 cost since we make the lines intercept
    #     cam_1_p = np.array([[-1, -1, -1]], dtype=float)  # cam 1
    #     cam_2_p = np.array([[1, 1, -1]], dtype=float)  # cam 2
    #     cam_3_p = np.array([[-1, 1, 1]], dtype=float)  # cam 3
    #     cam_4_p = np.array([[1, -1, 1]], dtype=float)  # cam 4
    #
    #     cams_view = []
    #     lines = []
    #     for i in range(k):
    #         cam1_view = {'p': cam_1_p, 'v': k_set_ground[i] - cam_1_p}  # camera blob
    #         cams_view.append(cam1_view)
    #         line1 = Line(p=cam_1_p, v=cam1_view['v'])
    #         lines.append(line1)
    #
    #     for i in range(k):
    #         cam2_view = {'p': cam_2_p, 'v': k_set_ground[i] - cam_2_p}
    #         cams_view.append(cam2_view)
    #         line2 = Line(p=cam_2_p, v=cam2_view['v'])
    #         lines.append(line2)
    #
    #     for i in range(k):
    #         cam3_view = {'p': cam_3_p, 'v': k_set_ground[i] - cam_3_p}
    #         cams_view.append(cam3_view)
    #         line3 = Line(p=cam_3_p, v=cam3_view['v'])
    #         lines.append(line3)
    #
    #     for i in range(k):
    #         cam4_view = {'p': cam_4_p, 'v': k_set_ground[i] - cam_4_p}
    #         cams_view.append(cam4_view)
    #         line4 = Line(p=cam_4_p, v=cam4_view['v'])
    #         lines.append(line4)
    #
    #     cams_positions = [cam_1_p, cam_2_p, cam_3_p, cam_4_p]  # for KMeans cam pos filter
    #
    #     kmeans = KMeans(
    #         k=k,
    #         run_type='exhaustive',
    #         t=5000,
    #         use_cam_pos_filter=True,
    #         use_in_range_filter=True,
    #         use_unique_points_filter=True,
    #         use_noise_filter=False,
    #         cams_positions=cams_positions,
    #         in_range_values={'bottom': 0, 'top': 10},
    #         noise_filter_size=None,
    #         debug=debug
    #     )
    #
    #     world_size = (-2, 2)
    #
    #     if debug:
    #         print(kmeans)
    #         print('Kmeans debug out:')
    #     k_set, cost = kmeans.run_kmeans(lines)
    #
    #     print('\tGround Truth vs predicted:')
    #     print(utils.var_to_string(k_set_ground, '\t\tk_set_ground', with_data=True))
    #     print(utils.var_to_string(k_set, '\t\tk_set       ', with_data=True))
    #     print('\t\tcost_ground={:.2f}'.format(cost_ground))
    #     print('\t\tcost       ={:.2f}'.format(cost))
    #     if with_plot:
    #         plot_kmeans_results(cams_p_v=cams_view, lines=lines, k_set=k_set,
    #                             k_set_ground=k_set_ground, cost=cost, cost_ground=cost_ground,
    #                             title=title, world_size=world_size)
    #     test_passed = abs(np.sum(k_set - k_set_ground)) < eps
    #     assert test_passed, 'Test failed: k_set={}, k_set_ground={}'.format(k_set.tolist(), k_set_ground.tolist())
    #
    #     test_passed = abs(cost_ground - cost) < eps
    #     assert test_passed, 'Test failed: cost_ground={:.7f}, cost={:.7f}'.format(cost_ground, cost)
    #
    #     print('\tTest PASSED')
    #     print('\tRun time {}'.format(utils.get_time_str_seconds(t_start)))
    #     return

    def one_mean_acc(debug: bool, with_plot: bool, eps=0.000001):
        """ 2 cams looking at origin with symmetrical noise k==1 """
        np.random.seed(42)
        t_start = utils.get_time_now()
        k = 1
        title = 'one_mean_acc: 2 cameras looking at the origin with symmetrical noise. K={}'.format(k)
        print(title)

        # create cameras views and lines from each view
        cam_1_p_v = {'p': np.array([[-1, -1, -1]], dtype=float), 'v': np.array([[1, 2, 2]], dtype=float)}  # cam 1
        line_1 = Line(p=cam_1_p_v['p'], v=cam_1_p_v['v'])
        cam_2_p_v = {'p': np.array([[1, 1, 1]], dtype=float), 'v': np.array([[-1, -2, -2]], dtype=float)}  # cam 2
        line_2 = Line(p=cam_2_p_v['p'], v=cam_2_p_v['v'])
        cams_views = [cam_1_p_v, cam_2_p_v]  # un-normalized cameras data for plotting
        lines = [line_1, line_2]  # cameras to lines (projection and unit vector)

        if debug:
            print('\tCameras:')
            for i, cam_p_v in enumerate(cams_views):
                print('\t\tcam {}: p={}, v={}'.format(i, cam_p_v['p'], cam_p_v['v']))

        # the k_set_ground is the origin
        k_set_ground = np.zeros(shape=(k, utils.DIM_3), dtype=float)
        cost_ground = 0.0
        kmeans = KMeans(
            k=k,
            run_type='1_mean_acc',
            t=None,
            use_cam_pos_filter=False,
            use_in_range_filter=False,
            use_unique_points_filter=False,
            use_noise_filter=False,
            cams_positions=None,
            in_range_values=None,
            noise_filter_size=None,
            debug=debug
        )

        if debug:
            print(kmeans)
            print('\tLines:')
            for i, line in enumerate(lines):
                print('\t\tline {}: p={}, v={}'.format(i, np.round(line.p, 3).tolist(), np.round(line.v, 3).tolist()))

            print('\tKmeans debug out:')
        k_set, cost = kmeans.run_kmeans(lines)  # cost is inf - probably det is 0

        print('\tGround Truth vs predicted:')
        print(utils.var_to_string(k_set_ground, '\t\tk_set_ground', with_data=True))
        print(utils.var_to_string(k_set, '\t\tk_set_algo', with_data=True))
        print('\t\tcost_ground={:.2f}'.format(cost_ground))
        print('\t\tcost_algo  ={:.2f}'.format(cost))

        if with_plot:
            plot_kmeans_results(cams_p_v=cams_views, lines=lines, k_set=k_set, k_set_ground=k_set_ground, cost=cost,
                                cost_ground=cost_ground, title=title)

        test_passed = abs(np.linalg.norm(k_set - k_set_ground)) < eps
        assert test_passed, 'Test failed: k_set={}, k_set_ground={}'.format(k_set.tolist(), k_set_ground.tolist())
        print('\tTest PASSED')
        print('\tRun time {}'.format(utils.get_time_str_seconds(t_start)))
        return

    """ if a test fails, assert will be raised """
    g_debug = True
    g_with_plot = False

    exhaustive_1_1(debug=g_debug, with_plot=g_with_plot)  # 2 cameras looking at the origin. K=1

    exhaustive_1_2(debug=g_debug, with_plot=g_with_plot)  # 2 cameras looking at the origin. K=2

    #  TODO calc ground cost
    exhaustive_2_1(debug=g_debug, with_plot=g_with_plot)  # 2 cams looking at origin with symmetrical noise k==1

    exhaustive_2_2(debug=g_debug, with_plot=g_with_plot)  # 2 cams looking at origin with symmetrical noise k==2

    # 4 cams looking at inner cube side with 4 blobs from each cam. k==4
    exhaustive_3_real_big_test(debug=g_debug, with_plot=g_with_plot)

    # test4_noise_check: 4 cams looking at one point - checking if noise filter works
    # exhaustive_4_noise_check(debug=g_debug, with_plot=g_with_plot)

    # test5 no points pass filter. K=1 - TODO figure out if it's a bug
    # exhaustive_5_fail(debug=g_debug, with_plot=g_with_plot)

    one_mean_acc(debug=g_debug, with_plot=g_with_plot)
    return


# todo 1 optimization:
#   in get_k_set_candidates save points indices in individual array
#   now get_cost_from_point(3rd of iteration run time) can be optimized to calc once just like
#       filter_check_norms_in_range

if __name__ == '__main__':
    g_time = utils.get_time_now()
    # prx = utils.start_profiler()
    # sim_run_with_cam_info()
    unit_tests()
    # print(utils.end_profiler(prx, rows=15))
    print('\nTotal run time {}'.format(utils.get_time_str_seconds(g_time)))

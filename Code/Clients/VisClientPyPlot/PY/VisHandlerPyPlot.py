# code based on:
# https://www.galaxysofts.com/new/python-creating-a-real-time-3d-plot/
# worth checking
# https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations
# rotation:
# https://matplotlib.org/gallery/mplot3d/rotate_axes3d.html
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SharedCode.PY.utilsScript as utils
from SharedCode.PY.recordedDataLoader import load_algo_client_output_data
from Clients.Client.PY.ClientClass import Client
import matplotlib

matplotlib.use('TkAgg')

PROJECT_HOME = '../../../'  # relative to VisMain.py
PROJECT_ABS_PATH = os.path.abspath(PROJECT_HOME)
print('PROJECT_ABS_PATH: {}'.format(PROJECT_ABS_PATH))
os.chdir(PROJECT_ABS_PATH)

VIS_CONSTANTS_PATH = 'Clients/VisClientPyPlot/VisConstantsPyPlot.json'

CFG_FILES = [utils.SHARED_CONSTANTS_PATH, VIS_CONSTANTS_PATH]
for cfg_file in CFG_FILES:
    assert os.path.exists(cfg_file), 'cant find {}'.format(cfg_file)
assert os.path.exists(utils.SERVER_ADDRESS_PATH), 'cant find {}'.format(utils.SERVER_ADDRESS_PATH)


class VisHandlerPyPlot(Client):
    def __init__(self):
        np.random.seed(42)
        print('VisClient::VisClient()')
        cfg = utils.load_jsons(CFG_FILES, ack=False)
        server_json = utils.load_json(utils.SERVER_ADDRESS_PATH, ack=False)

        super().__init__(
            cfg=cfg,
            local_mode=cfg['misc_vis']['local_mode_cfg']['local_mode'],
            send_output_data_from_ds=False,  # VIS cant 'send fake data'. it just sends done
            server_json=server_json,
            name='VisClientPyPlot',
        )
        assert 'calib_data' in cfg, 'No calib data found on cfg'  # Vis cant run without a calib file
        # init all members
        self.rps_playing = []  # given from server after connection established
        self.algo_scatter = None  # needed to update data each k_set received
        self.center_mass_label = None  # center mass of k_set coordinates
        self.center_mass_label_base = None
        self.loaded_input_data, self.input_ds_size = None, None  # in data variables
        self.axes = None
        self.fig_canvas = None
        self.arrows_just_cams = None
        self.cameras_params = None
        self.data_arrows_cfg = None
        self.local_block_last = False
        self.max_iters = None
        return

    def initialize(self):
        misc_vis = self.cfg['misc_vis']
        print('Extra data post server ctor')
        self.do_work(iteration_t='shake msg 4/5', expected_msg='post_ctor_server')
        print('Playing rps meta data:')
        self.cameras_params = utils.CameraParams.read_calib_data(self.cfg['calib_data'], self.rps_playing, self.debug)
        if self.local_mode:
            self.loaded_input_data, self.input_ds_size = load_algo_client_output_data(
                self.ds_path, self.detection_mode)
            self.local_block_last = misc_vis['local_mode_cfg']['block_last_iter']
            if self.local_block_last:
                self.max_iters = misc_vis['local_mode_cfg']['max_iters']

        print('m_run_mode={}'.format(self.run_mode))
        print('m_local_mode={}'.format(self.local_mode))
        if self.local_mode:
            print('\tds size={} (from 1 to ds_size)'.format(self.input_ds_size))
        print('m_detection_mode={}'.format(self.detection_mode))

        if misc_vis['block_post_init']:
            self.init_plot(self.cameras_params, block=True)
        self.init_plot(self.cameras_params, block=False)
        return

    def init_plot(self, cameras_params: dict, block: bool) -> None:
        misc_vis = self.cfg['misc_vis']
        fig = plt.figure()
        utils.move_plot(fig, where=misc_vis['3d_plot_window_location'])
        self.fig_canvas = fig.canvas
        self.axes = Axes3D(fig)
        self.axes.autoscale(enable=True, axis='both', tight=True)

        # Setting the axes properties
        self.axes.set_facecolor(misc_vis['face_color'])
        world_bot, world_top = misc_vis['world_size']['bottom'], misc_vis['world_size']['top']
        self.axes.set_xlim3d([world_bot, world_top])
        self.axes.set_ylim3d([world_bot, world_top])
        self.axes.set_zlim3d([world_bot, world_top])

        ticks_list = []
        jump = int((world_top - world_bot) / 4)
        for i in range(world_bot, world_top + 1, jump):
            ticks_list.append(i)
        self.axes.set_xticks(ticks_list)
        self.axes.set_yticks(ticks_list)
        self.axes.set_zticks(ticks_list)

        label_color = misc_vis['label_color']
        self.axes.set_xlabel("X", color=label_color)
        self.axes.set_ylabel("Y", color=label_color)
        self.axes.set_zlabel("Z", color=label_color)

        self.axes.grid(False)

        # background_and_alpha = (0.5, 0.2, 0.5, 0.4)  # RGB and alpha
        # background_and_alpha = matplotlib.colors.to_rgba('lightgrey', alpha=None)
        # self.axes.w_xaxis.set_pane_color(background_and_alpha)
        # self.axes.w_yaxis.set_pane_color(background_and_alpha)
        # self.axes.w_zaxis.set_pane_color(background_and_alpha)

        # k starting points in 3d
        origin_k_3d_points = np.zeros(shape=(1, utils.DIM_3))
        xs, ys, zs = origin_k_3d_points[:, 0], origin_k_3d_points[:, 1], origin_k_3d_points[:, 2]

        # setting algo client data plot
        algo_cfg = self.cfg['algo_data_scatter']
        self.data_arrows_cfg = algo_cfg['arrows']
        self.algo_scatter = self.axes.scatter(xs, ys, zs, c=algo_cfg['color'], s=algo_cfg['marker_size'],
                                              label=algo_cfg['label'])

        # setting label center mass
        center_mass_label_cfg = self.cfg['center_mass_label']
        if center_mass_label_cfg['show_label']:
            self.center_mass_label_base = center_mass_label_cfg['label_base']
            self.center_mass_label = self.axes.text2D(center_mass_label_cfg['x1'], center_mass_label_cfg['y1'],
                                                      center_mass_label_cfg['label_base'].format(np.zeros(3)),
                                                      transform=self.axes.transAxes, color=algo_cfg['color'])

        cube_cfg = misc_vis['cube']
        if cube_cfg['add_cube']:
            utils.add_cube(self.axes, cube_cfg['edge_len'], cube_cfg['add_corners_labels'])

        # setting cameras on the plot
        cams_cfg = self.cfg['cams_scatter']

        if cams_cfg['draw_other_ids']:  # each cam with mac/port
            MAC_PORT_MIN_CHARS = 4  # TODO to cfg
            color_map = utils.get_color_map(n=len(cameras_params))
            for i, camera_params in enumerate(cameras_params):
                label = camera_params.get_id(self.mac_p_del)[-MAC_PORT_MIN_CHARS:]
                marker_size = cams_cfg['marker_size']
                if np.array_equal(camera_params.p, [0., 0., 0.]):  # normal cam
                    label = '{}(G)'.format(label)
                    marker_size *= 2
                xs, ys, zs = camera_params.p
                self.axes.scatter(xs, ys, zs, color=color_map[i], s=marker_size, label=label)
        else:  # one label to all other cams
            marker_size = cams_cfg['marker_size']
            other_cams, id_gr = [], None
            for i, camera_params in enumerate(cameras_params):
                if not np.array_equal(camera_params.p, [0., 0., 0.]):  # normal cam
                    other_cams.append(camera_params.p)
                else:
                    id_gr = '{}(G)'.format(camera_params.get_id(self.mac_p_del))
            # draw all cams except ground once
            other_cams = np.array(other_cams, dtype=float)
            xs, ys, zs = other_cams[:, 0], other_cams[:, 1], other_cams[:, 2]
            self.axes.scatter(xs, ys, zs, color=cams_cfg['other_cams_color'], s=marker_size, label='All cameras')
            # add ground (marker twice as big and with a label)
            xs, ys, zs = origin_k_3d_points[:, 0], origin_k_3d_points[:, 1], origin_k_3d_points[:, 2]
            self.axes.scatter(xs, ys, zs, c=cams_cfg['ground_cam_color'], s=marker_size * 2, label=id_gr)

        arrow_cfg = cams_cfg["arrows"]
        if arrow_cfg['with_arrows']:
            utils.add_orientation_arrows_to_points(self.axes, cameras_params, arrow_cfg['color'], arrow_cfg['length'])
            self.arrows_just_cams = self.axes.artists[:]

        pyramid_cfg = cams_cfg["pyramids"]
        if pyramid_cfg['with_pyramids']:
            utils.add_orientation_pyramids_to_cams_pyplot(
                self.axes,
                cameras_params,
                pyramid_cfg['color'],
                pyramid_cfg['dist'],
                pyramid_cfg['height'],
                pyramid_cfg['width'])

        plt.legend(loc='upper right', ncol=1, fancybox=True, framealpha=0.5)

        if misc_vis['view'] == "cameras_to_object":
            # self.axes.view_init(azim=-97.0, elev=-45.0)  # for calib3 chessboard
            self.axes.view_init(azim=-85.0, elev=179.0)  # for 2021_04_28_14_10_41_8_cams_jac chessboard
        elif misc_vis['view'] == "object_to_cameras":
            self.axes.view_init(azim=90.0, elev=100.0)
        elif misc_vis['view'] == "top":
            self.axes.view_init(azim=-91.0, elev=-24.0)
        # re-render
        plt.draw()
        plt.show(block=block)
        plt.pause(0.0001)
        return

    def get_fake_msg_from_server(self, t: int, expected_msg: int) -> (str, int):
        j_in = {}

        if expected_msg == 'work':
            data_dict = self.loaded_input_data[t % self.input_ds_size]
            j_in = {'name': 'Server', 'msg': 'work', 'data': data_dict, 't': t}
        elif expected_msg == 'post_ctor_server':
            all_rp_macs = list(self.cfg["calib_data"].keys())
            assert len(all_rp_macs) > 0, "no rps found cfg['calib_data']"
            print('\t\tMac rps found on cfg[\'calib_data\']: {}'.format(all_rp_macs))
            max_rps = min(self.cfg['misc_vis']['local_mode_cfg']["max_rps"],
                          len(all_rp_macs))  # max `len(all_rp_macs)` rps
            rps_playing = all_rp_macs[:max_rps]  # macs of rps that will connect
            extra = {'rps_playing': rps_playing}
            j_in = {'name': 'Server', 'msg': 'post_ctor_server', 'extra': extra, 't': t}
            print('\t\tself.rps_playing: {}'.format(rps_playing))

        fake_data_in = utils.json_to_string(j_in)
        return fake_data_in, len(fake_data_in)

    def do_client_specific_work(self, j_in: json) -> json:
        j_out = {}
        if j_in['msg'] == 'work':
            self.handle_work(j_in)
            j_out = {'name': self.name, 'msg': 'done'}  # prepare output
        elif j_in['msg'] == 'post_ctor_server':
            self.rps_playing = j_in['extra']['rps_playing']
            extra_out = {}  # nothing to send
            j_out = {'name': self.name, 'msg': 'post_ctor_client', 't': 'shake msg 5/5', 'extra': extra_out}
        return j_out

    def handle_work(self, j_in: json) -> None:
        t = j_in['t']
        data_algo_dict = j_in['data']
        data_algo = np.array(data_algo_dict['d3points'])
        colors_per_point = None if 'colors' not in data_algo_dict else data_algo_dict['colors']
        lines = None if 'lines' not in data_algo_dict else data_algo_dict['lines']

        if self.debug:
            print(utils.var_to_string(data_algo, '\t\tdata_algo', with_data=True))
            if colors_per_point is not None:
                print(utils.var_to_string(colors_per_point, '\t\tcolors', with_data=True))
            if lines is not None:
                lines_str = '\t\tlines:'
                for line_dict in lines:
                    lines_str += 'p={}, v={}, '.format(line_dict['p'], line_dict['v'])
                print(lines_str)

        if data_algo.shape[0] <= 0:
            print(utils.WARNING.format('no data received on iter {}'.format(t), self.name))
            # noinspection PyProtectedMember
            data_algo = np.array(self.algo_scatter._offsets3d).T  # assign last iter data
            lines, colors_per_point = None, None

        if self.local_block_last and (t == self.max_iters - 1):  # block requested on last iter
            self.update_data(t, data_algo, lines, colors_per_point, block=True)
        else:
            if self.cfg['misc_vis']['block_post_iter']:
                self.update_data(t, data_algo, lines, colors_per_point, block=True)
                self.init_plot(self.cameras_params, block=False)
            self.update_data(t, data_algo, lines, colors_per_point, block=False)
        return

    def update_data(self, t: int, data_algo: np.array, lines, colors: list, block: bool) -> None:
        """
        :param t: iteration t
        :param data_algo: Kx3 np.array points. len(data_algo)>0
        :param lines: optional - the lines that was generated by kmeans
        :param colors: optional - colors list. size == 1 (same color for all k_set) or size == K (color per each p)
        :param block: draw and block
        """
        self.fig_canvas.set_window_title('iteration {}'.format(t))

        if self.center_mass_label_base is not None:
            # if got points: mean is center mass, else we have 1 point which is the center mass
            if data_algo.shape[0] > 1:
                center_mass_3dpoint = np.round(np.mean(data_algo, axis=0), self.float_pre)
            else:
                center_mass_3dpoint = data_algo
            self.center_mass_label.set_text(self.center_mass_label_base.format(center_mass_3dpoint))

        # update k_set new coordinates
        # data = np.max(data) - (data - np.min(data))
        # data_algo[:, 0] = np.max(data_algo[:, 0]) - (data_algo[:, 0] - np.min(data_algo[:, 0]))
        self.algo_scatter._offsets3d = data_algo.T

        if self.data_arrows_cfg['show'] and lines is not None:
            cameras_params = []
            for line_dict in lines:
                camera_params = utils.CameraParams(
                    mac=None,
                    port=None,
                    p=line_dict['p_orig'],
                    v=line_dict['v'],
                    rotation_mat=None,
                    intrinsic_mat=None,
                    distortion=None)
                cameras_params.append(camera_params)
            self.axes.artists = self.arrows_just_cams[:]
            utils.add_orientation_arrows_to_points(self.axes, cameras_params, arrow_color=self.data_arrows_cfg['c'],
                                                   arrow_len=self.data_arrows_cfg['len'])

        if colors is not None:
            self.algo_scatter._facecolor3d = colors
            self.algo_scatter._edgecolor3d = colors

        # print('azim', self.axes.azim)
        # print('elev', self.axes.elev)

        # re-render
        plt.draw()
        plt.show(block=block)
        plt.pause(0.0001)
        return

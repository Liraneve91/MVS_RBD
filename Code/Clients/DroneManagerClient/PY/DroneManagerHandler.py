import json
import os
import threading
from time import sleep
from threading import Thread
import asyncio
import numpy as np
import SharedCode.PY.utilsScript as utils
from Clients.Client.PY.ClientClass import Client
from Clients.DroneManagerClient.PY.DroneManager import DroneManager
from SharedCode.PY.recordedDataLoader import load_algo_client_output_data

PROJECT_HOME = '../../../'  # relative to OutMain.py
PROJECT_ABS_PATH = os.path.abspath(PROJECT_HOME)
print('PROJECT_ABS_PATH: {}'.format(PROJECT_ABS_PATH))
os.chdir(PROJECT_ABS_PATH)

OUT_CONSTANTS_PATH = 'Clients/OutClient/OutConstants.json'

CFG_FILES = [utils.SHARED_CONSTANTS_PATH, OUT_CONSTANTS_PATH]
for cfg_file in CFG_FILES:
    assert os.path.exists(cfg_file), 'cant find {}'.format(cfg_file)
assert os.path.exists(utils.SERVER_ADDRESS_PATH), 'cant find {}'.format(utils.SERVER_ADDRESS_PATH)

SLIDING_WINDOW_P_O_SIZE = 3
SLIDING_WINDOW_P_O_SIZE_INIT = 10
POS, NEG = 'positive', 'negative'
FW, BW, L, R, U, D = 'forward', 'backward', 'left', 'right', 'up', 'down'
IS_USE_SCRATCH = False


class Op:
    def __init__(self, op_type: str, num: float):
        self.op_type = op_type
        self.num = num
SIDE_LEN = 150
COMMANDS = [
    Op(BW, SIDE_LEN-50),
    Op(FW, SIDE_LEN+50),
    Op(BW, SIDE_LEN),
    Op(FW, SIDE_LEN),
    Op(L, SIDE_LEN/3),
    Op(BW, SIDE_LEN),
    Op(R, SIDE_LEN/3),
    Op(FW, SIDE_LEN),
    # Op(U, SIDE_LEN / 2),
    # Op(FW, SIDE_LEN / 2),
    # Op(BW, SIDE_LEN / 2),
]

class DroneManagerHandler(Client):
    def __init__(self):
        np.random.seed(42)
        print('OutClient::OutClient()')
        cfg = utils.load_jsons(CFG_FILES, ack=False)
        server_json = utils.load_json(utils.SERVER_ADDRESS_PATH, ack=False)

        super().__init__(
            cfg=cfg,
            local_mode=cfg['misc_out']['local_mode_cfg']['local_mode'],
            send_output_data_from_ds=False,  # VIS cant 'send fake data'. it just sends done
            server_json=server_json,
            name='OutClient',
        )
        assert 'calib_data' in cfg, 'No calib data found on cfg'  # Vis cant run without a calib file
        # init all members
        self.rps_playing = []  # given from server after connection established
        self.loaded_input_data, self.input_ds_size = None, None  # in data variables
        self.cameras_params = None
        self.local_block_last = False
        self.max_iterations = None
        self.dm = DroneManager()
        self.dm.takeoff()
        self.last_positions, self.last_orientations = [], []
        side_len = 100
        if not IS_USE_SCRATCH:
            # self.the_commands = [
            #     Op(FW, side_len),
            #     Op(R, side_len),
            #     Op(BW, side_len),
            #     Op(L, side_len),
            #     Op(U, side_len / 2),
            #     Op(FW, side_len / 2),
            #     Op(R, side_len / 2),
            #     Op(BW, side_len / 2),
            #     Op(L, side_len / 2)
            # ]
            self.the_commands = COMMANDS


        return

    def initialize(self) -> None:
        misc_out = self.cfg['misc_out']
        print('Extra data post server ctor')
        self.do_work(iteration_t='shake msg 4/5', expected_msg='post_ctor_server')
        print('Playing rps meta data:')
        self.cameras_params = utils.CameraParams.read_calib_data(self.cfg['calib_data'], self.rps_playing, self.debug)
        if self.local_mode:
            self.loaded_input_data, self.input_ds_size = load_algo_client_output_data(
                self.ds_path, self.detection_mode)
            self.local_block_last = misc_out['local_mode_cfg']['block_last_iter']
            if self.local_block_last:
                self.max_iterations = misc_out['local_mode_cfg']['max_iters']
        print('m_run_mode={}'.format(self.run_mode))
        print('m_local_mode={}'.format(self.local_mode))
        if self.local_mode:
            print('\tds size={} (from 1 to ds_size)'.format(self.input_ds_size))
        print('m_detection_mode={}'.format(self.detection_mode))
        if IS_USE_SCRATCH:
            import websockets as websockets
            Thread(target=self.create_web_socket_position_server, daemon=True).start()
            Thread(target=self.create_web_socket_command_server, daemon=True).start()
        else:
            t = threading.Thread(target=self.dm.handle_commands, args=(self.the_commands,))
            t.start()
        return

    def create_web_socket_position_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        asyncio.get_event_loop().run_until_complete(
            websockets.serve(self.send_data_by_web_socket, 'localhost', 8765))
        asyncio.get_event_loop().run_forever()

    def create_web_socket_command_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        asyncio.get_event_loop().run_until_complete(
            websockets.serve(self.handle_scratch_stack_by_web_socket, 'localhost', 8764))
        asyncio.get_event_loop().run_forever()

    async def send_data_by_web_socket(self, websocket):
        while True:
            if self.dm is not None and self.dm.current_p is not None and self.dm.current_o is not None:
                out_json = {'p': list(self.dm.current_p), 'o': list(self.dm.current_o)}
                out_data = utils.json_to_string(out_json)
                await websocket.send(out_data)
            sleep(0.4)

    async def handle_scratch_stack_by_web_socket(self, websocket):
        translation_commands = ['forward', 'backward', 'left', 'right', 'up', 'down']
        while True:
            command_raw = await websocket.recv()
            print(command_raw)
            command = command_raw.split(" ")
            try:
                if "takeoff" in command_raw:
                    pass
                    self.dm.takeoff()
                elif "land" in command_raw:
                    self.dm.land()
                elif command[0] in translation_commands:
                    op = Op(command[0], int(command[1]))
                    self.dm.translate_drone(op)
            except Exception:
                await websocket.send("shutingDown")
            await websocket.send("ok")

    def get_fake_msg_from_server(self, t: int, expected_msg: int) -> (str, int):
        """
        if you want to run on local mode(good for debug), record a data set and change params in OutConstants.json
        :return: simulated data - in this case, pre calculated algo client output
        """
        if expected_msg == 'work':
            data_dict = self.loaded_input_data[t % self.input_ds_size]
            j_in = {'name': 'Server', 'msg': 'work', 'data': data_dict, 't': t}
        elif expected_msg == 'post_ctor_server':
            all_rp_macs = list(self.cfg["calib_data"].keys())
            assert len(all_rp_macs) > 0, "no rps found cfg['calib_data']"
            print('\t\tMac rps found on cfg[\'calib_data\']: {}'.format(all_rp_macs))
            max_rps = min(self.cfg['misc_out']['local_mode_cfg']["max_rps"],
                          len(all_rp_macs))  # max `len(all_rp_macs)` rps
            rps_playing = all_rp_macs[:max_rps]  # macs of rps that will connect
            extra = {'rps_playing': rps_playing}
            j_in = {'name': 'Server', 'msg': 'post_ctor_server', 'extra': extra, 't': t}
            print('\t\tself.rps_playing: {}'.format(rps_playing))
        else:
            j_in = {}
        fake_data_in = utils.json_to_string(j_in)
        return fake_data_in, len(fake_data_in)

    def do_client_specific_work(self, j_in: json) -> json:
        """
        here is where client base class pass the specific work to this client
        :param j_in: contains data and meta data(sender, time t, msg details and data dict)
        :return: j_out: in this case, at work msg: done so the server could start iter t+1
        """
        if j_in['msg'] == 'work':
            j_out = self.handle_work(j_in)
        elif j_in['msg'] == 'post_ctor_server':
            self.rps_playing = j_in['extra']['rps_playing']
            extra_out = {}  # nothing to send
            j_out = {'name': self.name, 'msg': 'post_ctor_client', 't': 'shake msg 5/5', 'extra': extra_out}
        else:
            j_out = {}
        return j_out

    def handle_work(self, j_in: json) -> dict:
        """
        unpacking the data and meta data and doing some work
        :return: out dict
        """
        t = j_in['t']
        data_algo_dict = j_in['data']
        if self.debug:
            msg = '\t\ttime={}: j_in keys = {}, data_algo_dict keys = {}'
            print(msg.format(t, list(j_in.keys()), list(data_algo_dict.keys())))

        points3d = np.array(data_algo_dict['d3points'])
        # colors_per_point = None if 'colors' not in data_algo_dict else data_algo_dict['colors']
        # lines = None if 'lines' not in data_algo_dict else data_algo_dict['lines']
        self.update_drone_P_O(points3d)  # optional colors and lines

        j_out = {'name': self.name, 'msg': 'done'}  # prepare output for server - done means go to next iter
        return j_out

    @staticmethod
    def get_P_and_O_from_markers(red_3d: np.array, blue_3d: np.array) -> (np.asarray, np.asarray):
        """
        The function gets two markers position in 3d space and computes the drone P & O
        @param red_3d:
        @param blue_3d:
        @return: P and O (position and orientation of the drone)
        """
        red = red_3d[0:2]
        blue = blue_3d[0:2]
        mean = (red + blue) / 2.0
        red_shifted, blue_shifted = red - mean, blue - mean
        red_shifted_normalized = red_shifted / np.linalg.norm(red_shifted, ord=2)
        blue_shifted_normalized = blue_shifted / np.linalg.norm(blue_shifted, ord=2)
        theta = np.pi / 4.0 + np.pi
        c, s = np.cos(theta), np.sin(theta)
        rotation_mat = np.array([[c, -s], [s, c]])
        red_shifted_normalized_rotated = rotation_mat.T @ red_shifted_normalized
        blue_shifted_normalized_rotated = rotation_mat.T @ blue_shifted_normalized
        print('red_shifted_normalized_rotated: ', red_shifted_normalized_rotated)
        print('blue_shifted_normalized_rotated: ', blue_shifted_normalized_rotated)
        orientation = blue_shifted_normalized_rotated
        z_val = (red_3d[2] + blue_3d[2]) / 2.0
        P = np.array([mean[0], mean[1], z_val])
        O = np.array([orientation[0], orientation[1], 0])
        # return P, O
        return P, np.array([-1, 0, 0])

    def update_drone_P_O(self, points3d: np.array) -> None:
        """
        Update the DroneManager instance P & O using filters (avg, etc.)
        :param points3d: the 3d points calculated by algo client
        :return:
        @rtype: None
        """
        # if len(points3d) == 2:
        if len(points3d) > 0:
            print('points3d: ', points3d, '. P: ', self.dm.current_p, '. O: ', self.dm.current_o)
            # colors_dist = np.linalg.norm(points3d[0] - points3d[1], ord=2)
            # print('colors_dist:', colors_dist)
            self.dm.is_drone_detected = True
            # red_3d, blue_3d = points3d[0], points3d[1]
            # P, O = self.get_P_and_O_from_markers(red_3d, blue_3d)
            P, O = points3d[-1], np.array([-1, 0, 0])
            self.last_positions.append(P)
            # self.last_orientations.append(O)
            if len(self.last_positions) > SLIDING_WINDOW_P_O_SIZE:
                del self.last_positions[0]
            if len(self.last_orientations) > SLIDING_WINDOW_P_O_SIZE:
                del self.last_orientations[0]
            if len(self.last_positions) == SLIDING_WINDOW_P_O_SIZE:
                self.dm.current_p = np.mean(np.asarray(self.last_positions), axis=0)
                # self.dm.current_o = np.mean(np.asarray(self.last_orientations), axis=0)
            # time.sleep(0.3)
        else:
            self.dm.is_drone_detected = False
        return

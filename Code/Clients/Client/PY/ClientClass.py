import abc
import glob
import json
import os
import socket
import sys

import SharedCode.PY.utilsScript as utils
from SharedCode.PY.utilsScript import WARNING


class Client:
    def __init__(self, cfg: dict, local_mode: bool, send_output_data_from_ds: bool,
                 server_json: dict, name: str, extra_dict: dict = None):
        self.cfg = cfg
        self.run_mode = cfg["misc_global"]["system_mode"]  # calibration or mapping
        err_msg = 'run mode {} doesn\'t exist'.format(self.run_mode)
        assert self.run_mode in ['calibration', 'mapping', 'recording_ds'], err_msg

        self.calib_file_base_path = os.path.join('SharedCode/Calib', cfg['misc_global']['calib_folder'])
        calib_files = glob.glob("{}/*.json".format(self.calib_file_base_path))
        assert len(calib_files) <= 1, 'more than 1 calib file found on {}'.format(self.calib_file_base_path)

        err_msg = 'No Calib file but run mode is not calibration. Can only create new calibration'
        if len(calib_files) == 0 and not self.run_mode == 'calibration':
            assert False, err_msg
        else:
            calib_data = utils.load_json(calib_files[0], ack=False)
            cfg_keys, calib_data_keys = len(cfg), len(calib_data)
            cfg.update(calib_data)
            assert cfg_keys + calib_data_keys == len(cfg), 'duplicated keys in calib data and cfg'

        self.send_output_data_from_ds = send_output_data_from_ds
        self.local_mode = local_mode
        self.ds_path = os.path.join('SharedCode/Calib',
                                    self.cfg['misc_global']['calib_folder'],
                                    self.cfg['misc_global']['calib_ds'])
        self.name = name
        self.debug = cfg["misc_global"]["debug_mode"]
        self.msg_end = cfg["misc_global"]["end_message_suffix"]
        self.float_pre = cfg["misc_global"]["float_precision"]
        self.buf_len = cfg["misc_global"]["buflen"]
        self.detection_mode = cfg["misc_global"]["detection_mode"]
        self.mac_p_del = cfg["misc_global"]["mac_port_delimiter"]
        err_msg = 'detection mode {} doesn\'t exist'.format(self.detection_mode)
        assert self.detection_mode in ['chessboard', 'blobs', 'orb', 'skeleton', 'heads', 'colors', 'persons',
                                       'motion', 'drone', 'drone2'], err_msg
        self.connect_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (server_json['ip'], server_json['port'])

        print("Python  Version {}".format(sys.version))
        print('Working dir: {}'.format(os.getcwd()))
        if extra_dict is not None:
            print('extra_dict: {}'.format(extra_dict))

        print("Connecting to server {}".format(server_address))

        if local_mode:
            self.connect_socket = None
        else:
            try:
                # self.sock.connect(('localhost', 10000))
                self.connect_socket.connect(server_address)
                print('\tConnected to {}'.format(server_address))
            except ConnectionRefusedError:
                assert False, 'No server is found on {}'.format(server_address)
        print('\tClient info: {}'.format(self))

        # shake
        j_out = {'name': self.name, 'msg': 'hello', 't': 'shake msg 1/5'}
        data_out = utils.json_to_string(j_out)
        data_out_size = len(data_out)
        if not self.local_mode:
            utils.send_msg(self.connect_socket, self.buf_len, data_out, self.msg_end)
        utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=2, prefix='OUT')

        if not self.local_mode:
            data_in, data_in_size = utils.receive_msg(self.connect_socket, self.buf_len, self.msg_end)
        else:  # fake shake
            j_in = {'name': 'Server', 'msg': 'wait', 't': 'shake msg 2/5'}
            data_in = utils.json_to_string(j_in)
            data_in_size = len(data_in)
        if data_in:
            j_in = utils.string_to_json(data_in)
            if j_in['msg'] == 'wait':
                utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=2, prefix='IN ')

                j_out = {'name': self.name, 'msg': 'acknowledge', 't': 'shake msg 3/5'}
                if extra_dict is not None:
                    j_out['extra'] = extra_dict
                data_out = utils.json_to_string(j_out)
                data_out_size = len(data_out)
                if not self.local_mode:
                    utils.send_msg(self.connect_socket, self.buf_len, data_out, self.msg_end)
                utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=2, prefix='OUT')
            else:
                print(WARNING.format(self.cfg['errors']['bad_msg'], 'Server'))
        else:
            print(WARNING.format(self.cfg['errors']['no_data'], 'Server'))
            raise ValueError('Server Down')
        return

    @abc.abstractmethod
    def initialize(self) -> None:
        print('abs method - should not get here')
        exit(-1)
        return

    @abc.abstractmethod
    def get_fake_msg_from_server(self, time_t: int, expected_msg: str) -> (str, int):
        """ the data that would arrive if client wasn't on local mode """
        print('abs method - should not get here')
        exit(-1)
        return '', 0

    @abc.abstractmethod
    def do_client_specific_work(self, j_in: json) -> json:
        print('abs method - should not get here')
        exit(-1)
        return {}

    def process_data_in(self, data_in: str, data_in_size: int) -> str:
        """ receive input, do work and create output string """
        j_in = utils.string_to_json(data_in)
        # utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=1, prefix='IN ')

        j_out = self.do_client_specific_work(j_in)

        # prepare output
        data_out = utils.json_to_string(j_out)
        data_out_size = len(data_out)
        # utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=1, prefix='OUT')
        return data_out

    def __str__(self) -> str:
        msg = 'name={}, local_mode={}, send_output_data_from_ds={}, buf_len={}, sock={}'
        return msg.format(self.name, self.local_mode, self.send_output_data_from_ds,
                          self.buf_len, self.connect_socket)

    def do_work(self, iteration_t: int, expected_msg: str):
        if self.local_mode:
            data_in, data_in_size = self.get_fake_msg_from_server(iteration_t, expected_msg)
        else:
            data_in, data_in_size = utils.receive_msg(self.connect_socket, self.buf_len, self.msg_end)

        if data_in:
            data_out = self.process_data_in(data_in, data_in_size)
            if not self.local_mode:
                utils.send_msg(self.connect_socket, self.buf_len, data_out, self.msg_end)
        else:
            print(WARNING.format(self.cfg['errors']['no_data'], 'Server'))
            raise ValueError('Server Down')
        return

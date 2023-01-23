# https://pymotw.com/2/socket/tcp.html
# threads https://stackoverflow.com/questions/10810249/python-socket-multiple-clients
# threads https://codezup.com/socket-server-with-multiple-clients-model-multithreading-python/
import abc
import sys
import os
import glob
import json
import socket
import numpy as np
import SharedCode.PY.utilsScript as utils
from Server.PY.Calibration.CalibAgentScript import CalibAgent
import pyttsx3

PROJECT_HOME = '../../'  # relative to ServerMain.py
PROJECT_ABS_PATH = os.path.abspath(PROJECT_HOME)
print('PROJECT_ABS_PATH: {}'.format(PROJECT_ABS_PATH))
os.chdir(PROJECT_ABS_PATH)

SERVER_CONSTANTS_PATH = 'Server/ServerConstants.json'

CFG_FILES = [utils.SHARED_CONSTANTS_PATH, SERVER_CONSTANTS_PATH]
for cfg_file in CFG_FILES:
    assert os.path.exists(cfg_file), 'cant find {}'.format(cfg_file)


class ServerHandler:
    def __init__(self):
        np.random.seed(42)
        self.cfg = utils.load_jsons(CFG_FILES, ack=False)
        calib_file_base_path = os.path.join('SharedCode/Calib', self.cfg['misc_global']['calib_folder'])
        calib_files = glob.glob("{}/*.json".format(calib_file_base_path))
        assert len(calib_files) <= 1, 'more than 1 calib file found on {}'.format(calib_file_base_path)
        if len(calib_files) > 0:
            calib_data = utils.load_json(calib_files[0], ack=False)
            cfg_keys, calib_data_keys = len(self.cfg), len(calib_data)
            self.cfg.update(calib_data)
            assert cfg_keys + calib_data_keys == len(self.cfg), 'duplicated keys in calib data and cfg'

        self.ds_path = os.path.join('SharedCode/Calib',
                                    self.cfg['misc_global']['calib_folder'],
                                    self.cfg['misc_global']['calib_ds'])
        ip_pc = utils.get_ipv4(tabs=0, ack=True)  # real server ip
        server_json = utils.load_json(utils.SERVER_ADDRESS_PATH, ack=True)
        ip_pc = server_json['ip']

        # if len(server_json) == 0 or ip_pc != server_json['ip']:
        #     server_json = {'ip': ip_pc, 'port': self.cfg['misc_global']['port']}
        #     utils.save_json(utils.SERVER_ADDRESS_PATH, server_json)

        print("Python  Version {}".format(sys.version))
        print('Working dir: {}'.format(os.getcwd()))
        self.run_mode = self.cfg["misc_global"]["system_mode"]  # calibration or mapping
        err_msg = 'run mode {} doesn\'t exist'.format(self.run_mode)
        assert self.run_mode in ['calibration', 'mapping', 'recording_ds'], err_msg

        self.rps_playing = []  # macs of rps that will connect
        self.name = 'Server'
        self.camAgents, self.algoAgent, self.visAgents = [], None, []
        self.msg_end = self.cfg['misc_global']['end_message_suffix']
        self.buf_len = self.cfg['misc_global']['buflen']

        if self.run_mode == 'calibration':
            self.calibrate(ip_pc, server_json)
            exit(0)
        elif self.run_mode == 'recording_ds':
            self.record_ds(ip_pc, server_json)
            exit(0)

        # Algo cant run without a calib file
        assert 'calib_data' in self.cfg, 'cant run server on regular mode without calib data'

        # read camClientMeta (rp mac to its camera info) - all we need is the mac field
        all_rp_macs = []

        for i, cam_mac in enumerate(self.cfg["calib_data"].keys()):
            all_rp_macs.append(cam_mac)

        assert len(all_rp_macs) > 0, "no rps found on CamClientMeta.json"
        print('Mac rps found on CamClientMeta.json: {}'.format(all_rp_macs))

        active_agents, max_rps = self.calc_agents(all_rp_macs)

        print('Starting Server:')

        if active_agents > 0:
            server_address = (ip_pc, server_json['port'])
            self.sock = utils.open_server(server_address, ack=True)
            self.connect_to_real_clients(active_agents, all_rp_macs, max_rps)
        self.connect_to_fake_clients(all_rp_macs, max_rps)

        print('Server::Server() - Connected to:')

        all_agents = []
        all_agents += self.camAgents
        all_agents.append(self.algoAgent)
        all_agents += self.visAgents

        for agent in all_agents:
            print('\t{}'.format(agent))
        self.send_pre_run_data(all_agents)
        return

    def __del__(self):
        print('Server::~Server() closing active connections')
        if hasattr(self, 'camAgents'):
            for camAgent in self.camAgents:
                if camAgent and camAgent.active and camAgent.sock is not None:
                    camAgent.sock.close()
        if hasattr(self, 'algoAgent'):
            if self.algoAgent and self.algoAgent.active and self.algoAgent.sock is not None:
                self.algoAgent.sock.close()
        if hasattr(self, 'visAgents'):
            for visAgent in self.visAgents:
                if visAgent and visAgent.active and visAgent.sock is not None:
                    visAgent.sock.close()
        if hasattr(self, 'sock'):
            if self.sock:
                self.sock.close()
        return

    def calc_agents(self, all_rp_macs):
        active_agents, fake_agents, max_rps = 0, 0, 0
        print('Agents meta data:')
        for agent_meta in self.cfg['agents_meta_data']:
            print('\t{}: {}'.format(agent_meta['name'], agent_meta))
            if agent_meta["name"] == 'CamClient':
                max_rps = min(agent_meta["devices"], len(all_rp_macs))  # max `len(all_rp_macs)` rps
                assert max_rps > 0, 'invalid selection - rps playing must be >0'
                if agent_meta["active"]:
                    active_agents += max_rps
                else:
                    fake_agents += max_rps
            else:
                if agent_meta["active"]:
                    active_agents += 1
                else:
                    fake_agents += 1
        total_agents = active_agents + fake_agents
        msg = 'Expecting {} clients(real={}, fake={}, rps={})'
        print(msg.format(total_agents, active_agents, fake_agents, max_rps))
        return active_agents, max_rps

    def connect_to_real_clients(self, active_agents, all_rp_macs, max_rps):
        accepted = 0
        while accepted < active_agents:
            print('\tWaiting for connection {}/{}:'.format(accepted + 1, active_agents))
            client_sock, client_address = self.sock.accept()
            print('\t\tconnected to client on address {}'.format(client_address))
            # waiting for hello msg
            data_in, data_in_size = utils.receive_msg(client_sock, self.buf_len, self.msg_end)
            if data_in:
                j_in = utils.string_to_json(data_in)
                if j_in['msg'] == 'hello':
                    client_name = j_in['name']
                    utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=3,
                                       prefix='IN from {}'.format(client_name))

                    j_out = {'name': self.name, 'msg': 'wait', 't': 'shake msg 2/5'}  # sending wait

                    data_out = utils.json_to_string(j_out)
                    data_out_size = len(data_out)
                    utils.send_msg(client_sock, self.buf_len, data_out, self.msg_end)
                    utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=3,
                                       prefix='OUT for {}'.format(client_name))

                    # waiting for ACKNOWLEDGE msg
                    data_in, data_in_size = utils.receive_msg(client_sock, self.buf_len, self.msg_end)
                    if data_in:
                        j_in = utils.string_to_json(data_in)
                        if j_in['msg'] == 'acknowledge':
                            agent_name = j_in['name']
                            utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=3,
                                               prefix='IN from {}'.format(agent_name))
                            agent = None
                            if agent_name == 'CamClient':
                                mac_rp = j_in['extra']['mac']
                                # check rp mac is registered in all_rp_macs (has meta data. e.g. position mat)
                                mac_has_meta = (mac_rp in all_rp_macs)
                                # check it didn't connect twice
                                new_mac = (mac_rp not in self.rps_playing)
                                # check we didn't pass the agent_meta["devices"]
                                under_limit = len(self.camAgents) < max_rps
                                if mac_has_meta and new_mac and under_limit:
                                    agent = CamAgent(
                                        cfg=self.cfg,
                                        name=agent_name,
                                        sock=client_sock,
                                        client_adr=client_address,
                                        active=True,
                                        mac=mac_rp,
                                        ds_path=None
                                    )
                                    self.camAgents.append(agent)
                                    self.rps_playing.append(mac_rp)
                                    print('\t\tCreated Agent: {}'.format(agent))
                                    accepted += 1
                                else:
                                    if not under_limit:
                                        msg = 'rp limit reached({}). rejected connection.'.format(max_rps)
                                        print(utils.WARNING.format(msg, self.name))
                                    elif not mac_has_meta:
                                        print(utils.WARNING.format('rp mac is not in CamClientMeta', self.name))
                                    else:
                                        print(utils.WARNING.format('rp mac already connected', self.name))
                            elif agent_name == 'AlgoClient':
                                agent = AlgoAgent(
                                    cfg=self.cfg,
                                    name=agent_name,
                                    sock=client_sock,
                                    client_adr=client_address,
                                    active=True,
                                    ds_path=None
                                )
                                self.algoAgent = agent
                                accepted += 1
                            elif agent_name in ['VisClientPyPlot', 'VisClientQt', 'OutClient']:
                                agent = VisAgent(
                                    cfg=self.cfg,
                                    name=agent_name,
                                    sock=client_sock,
                                    client_adr=client_address,
                                    active=True
                                )
                                self.visAgents.append(agent)
                                accepted += 1
                            print('\t\tCreated Agent: {}'.format(agent))
                        else:
                            print(utils.json_to_string(j_in))
                            print(utils.WARNING.format(self.cfg['errors']['bad_msg'], client_address))
                    else:
                        print(utils.WARNING.format(self.cfg['errors']['no_data'], client_address))
                else:
                    print(utils.json_to_string(j_in))
                    print(utils.WARNING.format(self.cfg['errors']['bad_msg'], client_address))
            else:
                print(utils.WARNING.format(self.cfg['errors']['no_data'], client_address))
        return

    def connect_to_fake_clients(self, all_rp_macs, max_rps):
        for meta in self.cfg['agents_meta_data']:
            if not meta['active']:
                if meta['name'] == 'CamClient':
                    # take first `max_rps` macs from all_rp_macs
                    self.rps_playing = all_rp_macs[:max_rps]
                    for i, rp_mac in enumerate(self.rps_playing):
                        name = "{}{}".format('CamClient', i + 1)
                        agent = CamAgent(
                            cfg=self.cfg,
                            name=name,
                            mac=rp_mac,
                            sock=None,
                            client_adr=(None, None),
                            active=False,
                            ds_path=self.ds_path
                        )
                        self.camAgents.append(agent)
                elif meta['name'] == 'AlgoClient':
                    agent = AlgoAgent(
                        cfg=self.cfg,
                        name='AlgoClient',
                        sock=None,
                        client_adr=(None, None),
                        active=False,
                        ds_path=self.ds_path
                    )
                    self.algoAgent = agent
                elif meta['name'] in ['VisClientPyPlot', 'VisClientQt', 'OutClient']:
                    agent = VisAgent(
                        cfg=self.cfg,
                        name=meta['name'],
                        sock=None,
                        client_adr=(None, None),
                        active=False
                    )
                    self.visAgents.append(agent)
        print('rps playing: {}'.format(self.rps_playing))
        return

    def send_pre_run_data(self, all_agents: list):
        print('Server::send_pre_run_data()')
        # sending wait
        for agent in all_agents:
            # e.g. extra_dict['bla'] = 'x'
            extra_dict = {}
            if agent.name == 'CamClient':
                pass  # nothing to pass
            elif agent.name == 'AlgoClient':
                extra_dict['rps_playing'] = self.rps_playing
            elif agent.name in ['VisClientPyPlot', 'VisClientQt', 'OutClient']:
                extra_dict['rps_playing'] = self.rps_playing
            t = 'shake msg 4/5'
            # send wait again with extra dict (params post connection)
            j_out = {'name': self.name, 'msg': 'post_ctor_server', 't': t, 'extra': extra_dict}
            data_out = utils.json_to_string(j_out)
            if agent.active:
                utils.send_msg(agent.sock, self.buf_len, data_out, self.msg_end)
            data_out_size = len(data_out)
            utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=1,
                               prefix='OUT for {}'.format(agent.name))

            # waiting for ACKNOWLEDGE msg
            if agent.active:
                data_in, data_in_size = utils.receive_msg(agent.sock, self.buf_len, self.msg_end)
            else:
                data_in, data_in_size = agent.receive_fake_data(t=t, expected_msg='post_ctor_client')

            if data_in:
                j_in = utils.string_to_json(data_in)
                if j_in['msg'] == 'post_ctor_client':
                    # IF in future client pass more extra data - save it in a member in agent
                    # e.g. agent.extra = j_in['extra']
                    utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=1,
                                       prefix='IN from {}'.format(j_in['name']))
                else:
                    print(utils.json_to_string(j_in))
                    print(utils.WARNING.format(self.cfg['errors']['bad_msg'], agent))
            else:
                print(utils.WARNING.format(self.cfg['errors']['no_data'], agent))
        return

    def ask_agents_data(self, t: int):
        print('\task_agents_data():')
        begin_time = utils.get_time_now()

        # send camAgents to work
        for agent in self.camAgents:
            j_out = {'name': self.name, 'msg': 'work', 't': t}
            data_out = utils.json_to_string(j_out)
            if agent.active:
                utils.send_msg(agent.sock, self.buf_len, data_out, self.msg_end)
            data_out_size = len(data_out)
            utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=2,
                               prefix='OUT for {}'.format(agent.name))
        # receive output from camAgents
        for agent in self.camAgents:
            if agent.active:
                data_in, data_in_size = utils.receive_msg(agent.sock, self.buf_len, self.msg_end)
            else:
                data_in, data_in_size = agent.receive_fake_data(t, 'work')

            if data_in:
                j_in = utils.string_to_json(data_in)
                if j_in['msg'] == 'output':
                    agent.last_received_data = j_in['data']
                    utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=2,
                                       prefix='In from {}'.format(agent.name))
                else:
                    print(utils.json_to_string(j_in))
                    print(utils.WARNING.format(self.cfg['errors']['bad_msg'], agent))
            else:
                print(utils.WARNING.format(self.cfg['errors']['no_data'], agent))
                agent.last_received_data = None

        print('\task_agents_data() in round {} took {}'.format(t, utils.get_time_str_seconds(begin_time)))
        return

    def analyze_data(self, t):
        print('\tanalyze_data():')
        begin_time = utils.get_time_now()
        # prepare mac to data map - K 2d points from each rp
        # e.g. row: "dc:a6:32:0f:4c:ca":[[25.189,14.019],[12.636,39.661],[43.328,24.902],[-39.762,-9.997]]
        macs_to_data = {}
        for cam_agent in self.camAgents:
            macs_to_data[cam_agent.mac] = cam_agent.last_received_data

        # send algo to work
        j_out = {'name': self.name, 'msg': 'work', 't': t, 'data': macs_to_data}
        data_out = utils.json_to_string(j_out)
        if self.algoAgent.active:
            utils.send_msg(self.algoAgent.sock, self.buf_len, data_out, self.msg_end)
        data_out_size = len(data_out)
        utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=2,
                           prefix='OUT for {}'.format(self.algoAgent.name))

        # receive algo output
        if self.algoAgent.active:
            data_in, data_in_size = utils.receive_msg(self.algoAgent.sock, self.buf_len, self.msg_end)
        else:
            data_in, data_in_size = self.algoAgent.receive_fake_data(t, 'work')

        if data_in:
            j_in = utils.string_to_json(data_in)
            if j_in['msg'] == 'output':
                self.algoAgent.last_received_data = j_in['data']
                utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=2,
                                   prefix='In from {}'.format(self.algoAgent.name))
            else:
                print(utils.json_to_string(j_in))
                print(utils.WARNING.format(self.cfg['errors']['bad_msg'], self.algoAgent))
        else:
            print(utils.WARNING.format(self.cfg['errors']['no_data'], self.algoAgent))
            self.algoAgent.last_received_data = None
        print('\tanalyze_data() in round {} took {}'.format(t, utils.get_time_str_seconds(begin_time)))
        return

    def visualize_object(self, t):
        print('\tvisualize_object():')
        begin_time = utils.get_time_now()

        # send visualizer to work
        data_dict_algo = self.algoAgent.last_received_data
        print(utils.var_to_string(data_dict_algo, '\t\tdata_dict_algo', with_data=False))
        # print('\t\t{}({}) data: {}'.format(self.algoAgent.name, self.algoAgent.code, data_dict_algo['d3points']))
        # if 'lines_dict' in data_dict_algo:
        #     print('\t\t\tlines: {}'.format(data_dict_algo['lines_dict']))

        j_out = {'name': self.name, 'msg': 'work', 'data': data_dict_algo, 't': t}

        data_out = utils.json_to_string(j_out)
        for visAgent in self.visAgents:
            if visAgent.active:
                utils.send_msg(visAgent.sock, self.buf_len, data_out, self.msg_end)
            data_out_size = len(data_out)
            utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=2,
                               prefix='OUT for {}'.format(visAgent.name))
        for visAgent in self.visAgents:
            # receive visualizer done
            if visAgent.active:
                data_in, data_in_size = utils.receive_msg(visAgent.sock, self.buf_len, self.msg_end)
            else:
                data_in, data_in_size = visAgent.receive_fake_data(t, 'work')

            if data_in:
                j_in = utils.string_to_json(data_in)
                if j_in['msg'] == 'done':
                    utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=2,
                                       prefix='In from {}'.format(visAgent.name))
                else:
                    print(utils.json_to_string(j_in))
                    print(utils.WARNING.format(self.cfg['errors']['bad_msg'], visAgent))
            else:
                print(utils.WARNING.format(self.cfg['errors']['no_data'], visAgent))
                self.algoAgent.last_received_data = None
        print('\tanalyze_data() in round {} took {}'.format(t, utils.get_time_str_seconds(begin_time)))
        return

    def calibrate(self, ip_pc, server_json):
        devices = -1
        print('Agents meta data:')
        for agent_meta in self.cfg['agents_meta_data']:
            if agent_meta["name"] == 'CamClient':
                print('\t{}: {}'.format(agent_meta["name"], agent_meta))
                devices = agent_meta["devices"] if agent_meta['active'] else -1

        calib_params = self.cfg['calibration']
        part1_params = calib_params['part1']
        part2_params = calib_params['part2']

        # either we have cams and we calibrate with taking pics or we use existing folder and skip part 1
        if devices > 0:
            self.connect_only_to_devices(ip_pc, server_json, devices)
            print('Calibrating with Real devices:')
            data_root_base = calib_params['data_root_base']
            data_root_suffix = utils.get_current_time_stamp()
            calib_folder = '{}/{}'.format(data_root_base, data_root_suffix)
            utils.create_dir_if_not_exists(calib_folder)
            calib_images_folder = '{}/CalibImages'.format(calib_folder)
            utils.create_dir_if_not_exists(calib_images_folder)
            self.calib_part1(calib_images_folder, part1_params, iterations=part1_params['iterations'])
            self.calib_part2(calib_folder, part2_params)

        elif devices == -1:
            print('Calibrating from a Folder:')
            data_root_base = calib_params['data_root_base']
            calib_folder = os.path.join(data_root_base, part2_params['data_folder'])
            calib_images_folder = '{}/CalibImages'.format(calib_folder)
            print('\tcalib_images_folder {}'.format(calib_images_folder))
            if os.path.exists(calib_images_folder):  # calib from existing folder
                self.calib_part2(calib_folder, part2_params)
            else:
                print(utils.WARNING.format('data_root {} doesnt exist'.format(calib_folder), 'Calib'))
        else:
            print(utils.WARNING.format('devices invalid value {}'.format(devices), 'Calib'))
        return

    def connect_only_to_devices(self, ip_pc, server_json, devices):
        print('Starting Server on CalibMode:')
        server_address = (ip_pc, server_json['port'])
        self.sock = utils.open_server(server_address, ack=True)
        accepted = 0
        while accepted < devices:
            print('\tWaiting for connection {}/{}:'.format(accepted + 1, devices))
            client_sock, client_address = self.sock.accept()
            print('\t\tconnected to client on address {}'.format(client_address))
            # waiting for hello msg
            data_in, data_in_size = utils.receive_msg(client_sock, self.buf_len, self.msg_end)
            if data_in:
                j_in = utils.string_to_json(data_in)
                if j_in['msg'] == 'hello':
                    client_name = j_in['name']
                    utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=3,
                                       prefix='IN from {}'.format(client_name))
                    # sending wait
                    j_out = {'name': self.name, 'msg': 'wait', 't': 'shake msg 2/5'}

                    data_out = utils.json_to_string(j_out)
                    data_out_size = len(data_out)
                    utils.send_msg(client_sock, self.buf_len, data_out, self.msg_end)
                    utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=3,
                                       prefix='OUT for {}'.format(client_name))

                    # waiting for ACKNOWLEDGE msg
                    data_in, data_in_size = utils.receive_msg(client_sock, self.buf_len, self.msg_end)
                    if data_in:
                        j_in = utils.string_to_json(data_in)
                        if j_in['msg'] == 'acknowledge':
                            agent_name = j_in['name']
                            utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=3,
                                               prefix='IN from {}'.format(agent_name))
                            if agent_name == 'CamClient':
                                mac_rp = j_in['extra']['mac']
                                if mac_rp not in self.rps_playing:  # check it didn't connect twice
                                    agent = CamAgent(cfg=self.cfg, name=agent_name, sock=client_sock,
                                                     client_adr=client_address, active=True, mac=mac_rp)
                                    self.camAgents.append(agent)
                                    self.rps_playing.append(mac_rp)
                                    print('\t\tCreated Agent: {}'.format(agent))
                                    accepted += 1
                                else:
                                    print(utils.WARNING.format('rp mac already connected', self.name))
                        else:
                            print(utils.json_to_string(j_in))
                            print(utils.WARNING.format(self.cfg['errors']['bad_msg'], client_address))
                    else:
                        print(utils.WARNING.format(self.cfg['errors']['no_data'], client_address))
                else:
                    print(utils.json_to_string(j_in))
                    print(utils.WARNING.format(self.cfg['errors']['bad_msg'], client_address))
            else:
                print(utils.WARNING.format(self.cfg['errors']['no_data'], client_address))

        print('Server::Server() - Connected to:')
        for cam_agent in self.camAgents:
            print('\t{}'.format(cam_agent))

        self.send_pre_run_data(self.camAgents.copy())
        return

    def calib_part1(self, calib_images_folder: str, part1_params: dict, iterations: int):
        begin = utils.get_time_now()
        print('Part 1: tell devices to take {} images and send them here'.format(iterations))

        for cam in self.camAgents:
            mac_no_colons = cam.mac.replace(':', '')
            utils.create_dir_if_not_exists('{}/{}'.format(calib_images_folder, mac_no_colons))

        print('First command devices to capture and save locally:')
        timer_secs = part1_params['timer']
        speech_engine = pyttsx3.init()

        for iteration_i in range(iterations):
            print('\tIteration {}/{}:'.format(iteration_i + 1, iterations))
            txt_to_speak = str(iteration_i + 1)
            speech_engine.say(txt_to_speak)
            speech_engine.runAndWait()
            utils.timer(seconds=timer_secs, action='Taking photo {}'.format(iteration_i), tabs=2)
            for cam in self.camAgents:
                j_out = {'name': 'CalibAgent', 'msg': 'calib_take_img', 't': iteration_i}
                data_out = utils.json_to_string(j_out)
                utils.send_msg(cam.sock, self.buf_len, data_out, self.msg_end)
                data_out_size = len(data_out)
                utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=2,
                                   prefix='OUT for {}'.format(cam.name))
            # receive image taken status
            for cam in self.camAgents:
                data_in, data_in_size = utils.receive_msg(cam.sock, self.buf_len, self.msg_end)
                if data_in:
                    j_in = utils.string_to_json(data_in)
                    if j_in['msg'] == 'calib_image_taken':
                        utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=2,
                                           prefix='In from {}'.format(cam.name))
                    else:
                        print(utils.json_to_string(j_in))
                        print(utils.WARNING.format(self.cfg['errors']['bad_msg'], cam))
                else:
                    print(utils.WARNING.format(self.cfg['errors']['no_data'], cam))
        print('Second get all images saved and store in server side:')
        # collect all images
        for iteration_i in range(iterations):
            print('\tIteration {}/{}'.format(iteration_i + 1, iterations))
            for cam in self.camAgents:
                j_out = {'name': 'CalibAgent', 'msg': 'calib_send_img', 't': iteration_i}
                data_out = utils.json_to_string(j_out)
                utils.send_msg(cam.sock, self.buf_len, data_out, self.msg_end)
                data_out_size = len(data_out)
                utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=2,
                                   prefix='OUT for {}'.format(cam.name))

            incoming_imgs_iter_i = []
            for cam in self.camAgents:
                data_in, data_in_size = utils.receive_msg(cam.sock, self.buf_len, self.msg_end)
                if data_in:
                    j_in = utils.string_to_json(data_in)
                    if j_in['msg'] == 'calib_output_img':
                        imgs_dict = j_in['data']
                        for cam_port, img_list in imgs_dict.items():
                            img = np.array(img_list, dtype='uint8')
                            incoming_imgs_iter_i.append(img)
                            img_dst_folder = '{}/{}/{}'.format(calib_images_folder, cam.mac.replace(':', ''), cam_port)
                            utils.create_dir_if_not_exists(img_dst_folder, ack=False)
                            full_path = '{}/{}.jpg'.format(img_dst_folder, iteration_i)

                            utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=2,
                                               prefix='In from {}'.format(cam.name))
                            utils.save_img(full_path, img, tabs=3)
                    else:
                        print(utils.json_to_string(j_in))
                        print(utils.WARNING.format(self.cfg['errors']['bad_msg'], cam))
                else:
                    print(utils.WARNING.format(self.cfg['errors']['no_data'], cam))
                    cam.last_received_data = None
            if part1_params['show_imgs_server_side']:
                ms = part1_params['show_imgs_server_side_ms']
                title = 'SERVER iter {} ({} images)'.format(iteration_i, len(incoming_imgs_iter_i))
                utils.display_open_cv_images(incoming_imgs_iter_i, ms=ms, title=title)
        print('part 1 took {}'.format(utils.get_time_str_seconds(begin)))
        return

    def calib_part2(self, calib_folder: str, part2_params: dict):
        print('Part 2: invoke calib module:')
        begin = utils.get_time_now()
        calib_agent = CalibAgent(
            calib_folder=calib_folder,
            float_pre=self.cfg['misc_global']['float_precision'],
            show_corners=part2_params['show_corners'],
            show_plot=part2_params['show_plot'],
            show_views_graph=part2_params['show_views_graph'],
            square_size=part2_params['square_size'],
            squares_num_width=part2_params['squares_num_width'],
            squares_num_height=part2_params['squares_num_height']
        )
        print(calib_agent)
        calib_agent.run_calibration()
        print('recalibrate() part 2 took {}'.format(utils.get_time_str_seconds(begin)))
        return

    def record_ds(self, ip_pc, server_json):
        devices = -1
        print('Agents meta data:')
        for agent_meta in self.cfg['agents_meta_data']:
            if agent_meta["name"] == 'CamClient':
                print('\t{}: {}'.format(agent_meta["name"], agent_meta))
                devices = agent_meta["devices"] if agent_meta['active'] else -1

        # either we have cams and we calibrate with taking pics or we use existing folder and skip part 1
        if devices > 0:
            self.connect_only_to_devices(ip_pc, server_json, devices)
            print('record_ds with RPS:')
            save_folder = self.cfg['ds_recording']['save_to']
            assert not os.path.exists(save_folder), '{} already exist. choose another name'.format(save_folder)
            utils.create_dir_if_not_exists(save_folder)
            self.record_ds_work(data_root=save_folder, iterations=self.cfg['ds_recording']['iterations'])
        else:
            print(utils.WARNING.format('devices invalid value {}'.format(devices), 'record_ds'))
        return

    def record_ds_work(self, data_root: str, iterations: int):
        begin = utils.get_time_now()
        print('Part 1: tell devices to take {} images and send them here'.format(iterations))

        for cam in self.camAgents:
            mac_no_colons = cam.mac.replace(':', '')
            utils.create_dir_if_not_exists('{}/{}'.format(data_root, mac_no_colons))

        print('First command devices to capture and save locally:')
        speech_engine = pyttsx3.init()

        for iteration_i in range(iterations):
            print('\tIteration {}/{}:'.format(iteration_i + 1, iterations))
            txt_to_speak = str(iteration_i + 1)
            speech_engine.say(txt_to_speak)
            speech_engine.runAndWait()
            # time.sleep(3)
            for cam in self.camAgents:
                j_out = {'name': 'CalibAgent', 'msg': 'calib_take_img', 't': iteration_i}
                data_out = utils.json_to_string(j_out)
                utils.send_msg(cam.sock, self.buf_len, data_out, self.msg_end)
                data_out_size = len(data_out)
                utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=2,
                                   prefix='OUT for {}'.format(cam.name))
            # receive image taken status
            for cam in self.camAgents:
                data_in, data_in_size = utils.receive_msg(cam.sock, self.buf_len, self.msg_end)
                if data_in:
                    j_in = utils.string_to_json(data_in)
                    if j_in['msg'] == 'calib_image_taken':
                        utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=2,
                                           prefix='In from {}'.format(cam.name))
                    else:
                        print(utils.json_to_string(j_in))
                        print(utils.WARNING.format(self.cfg['errors']['bad_msg'], cam))
                else:
                    print(utils.WARNING.format(self.cfg['errors']['no_data'], cam))
        print('Second get all images saved and store in server side:')
        # collect all images
        for iteration_i in range(iterations):
            print('\tIteration {}/{}'.format(iteration_i + 1, iterations))
            for cam in self.camAgents:
                j_out = {'name': 'CalibAgent', 'msg': 'calib_send_img', 't': iteration_i}
                data_out = utils.json_to_string(j_out)
                utils.send_msg(cam.sock, self.buf_len, data_out, self.msg_end)
                data_out_size = len(data_out)
                utils.buffer_print(utils.json_to_string(j_out), original_size=data_out_size, tabs=2,
                                   prefix='OUT for {}'.format(cam.name))

            incoming_imgs_iter_i = []
            for cam in self.camAgents:
                data_in, data_in_size = utils.receive_msg(cam.sock, self.buf_len, self.msg_end)
                if data_in:
                    j_in = utils.string_to_json(data_in)
                    if j_in['msg'] == 'calib_output_img':
                        imgs_dict = j_in['data']
                        for cam_port, img_list in imgs_dict.items():
                            img = np.array(img_list, dtype='uint8')
                            incoming_imgs_iter_i.append(img)
                            img_dst_folder = '{}/{}/{}'.format(data_root, cam.mac.replace(':', ''), cam_port)
                            utils.create_dir_if_not_exists(img_dst_folder, ack=False)
                            full_path = '{}/{}.jpg'.format(img_dst_folder, iteration_i)

                            utils.buffer_print(utils.json_to_string(j_in), original_size=data_in_size, tabs=2,
                                               prefix='In from {}'.format(cam.name))
                            utils.save_img(full_path, img, tabs=3)
                    else:
                        print(utils.json_to_string(j_in))
                        print(utils.WARNING.format(self.cfg['errors']['bad_msg'], cam))
                else:
                    print(utils.WARNING.format(self.cfg['errors']['no_data'], cam))
                    cam.last_received_data = None
            if self.cfg['ds_recording']['show_imgs_server_side']:
                ms = self.cfg['ds_recording']['show_imgs_server_side_ms']
                title = 'SERVER iter {} ({} images)'.format(iteration_i, len(incoming_imgs_iter_i))
                utils.display_open_cv_images(incoming_imgs_iter_i, ms=ms, title=title)
        print('record_ds_work took {}'.format(utils.get_time_str_seconds(begin)))
        return


class Agent:
    def __init__(self, cfg: json, name: str, sock: socket, client_adr: tuple, active: bool):
        self.cfg = cfg
        self.name = name
        self.sock = sock
        self.client_adr = client_adr
        self.active = active
        self.last_received_data = None  # keep data received from client here
        self.float_pre = cfg["misc_global"]["float_precision"]
        return

    def __str__(self) -> str:
        msg = 'name={}, active={}, client_adr={}'
        return msg.format(self.name, self.active, self.client_adr)

    @abc.abstractmethod
    def receive_fake_data(self, t: int, expected_msg: int) -> (str, int):
        print('abs')
        exit(-1)
        return '', 0


class CamAgent(Agent):
    def __init__(self, cfg: json, name: str, mac: str, sock: socket = None,
                 client_adr: tuple = (None, None), active: bool = False, ds_path: str = None):
        super().__init__(cfg, name, sock, client_adr, active)
        from SharedCode.PY.recordedDataLoader import load_cam_client_output_data
        self.mac = mac
        self.loaded_output_data, self.out_ds_size = None, None
        if not self.active:
            self.ds_path = ds_path
            detection_mode = self.cfg["misc_global"]["detection_mode"]
            self.loaded_output_data, self.out_ds_size = load_cam_client_output_data(ds_path, detection_mode, self.mac)
        return

    def __str__(self) -> str:
        msg = 'name={}, mac={}, active={}, client_adr={}'
        msg = msg.format(self.name, self.mac, self.active, self.client_adr)
        if not self.active:
            ds_full = os.path.join(self.ds_path, 'CamClientOutput', self.cfg["misc_global"]["detection_mode"],
                                   self.mac.replace(':', ''))
            msg += ', data_path(size {}): {}'.format(self.out_ds_size, ds_full)
        return msg

    def receive_fake_data(self, t: int, expected_msg: int) -> (str, int):
        j_in = {}
        if expected_msg == 'work':
            data_dict = self.loaded_output_data[t % self.out_ds_size]
            j_in = {'name': self.name, 'msg': 'output', 'data': data_dict}
        elif expected_msg == 'post_ctor_client':
            j_in = {'name': self.name, 'msg': 'post_ctor_client', 'extra': {}}
        data_in = utils.json_to_string(j_in)
        return data_in, len(data_in)


class AlgoAgent(Agent):
    def __init__(self, cfg: json, name: str, sock: socket = None,
                 client_adr: tuple = (None, None), active: bool = False, ds_path: str = None):
        super().__init__(cfg, name, sock, client_adr, active)
        from SharedCode.PY.recordedDataLoader import load_algo_client_output_data
        self.loaded_output_data, self.out_ds_size = None, None
        if not self.active:
            self.ds_path = ds_path
            detection_mode = self.cfg["misc_global"]["detection_mode"]
            self.loaded_output_data, self.out_ds_size = load_algo_client_output_data(ds_path, detection_mode)
        return

    def receive_fake_data(self, t: int, expected_msg: int) -> (str, int):
        j_in = {}
        if expected_msg == 'work':
            data_dict = self.loaded_output_data[t % self.out_ds_size]
            j_in = {'name': self.name, 'msg': 'output', 'data': data_dict}
        elif expected_msg == 'post_ctor_client':
            j_in = {'name': self.name, 'msg': 'post_ctor_client', 'extra': {}}
        data_in = utils.json_to_string(j_in)
        return data_in, len(data_in)

    def __str__(self) -> str:
        msg = 'name={}, active={}, client_adr={}'
        msg = msg.format(self.name, self.active, self.client_adr)
        if not self.active:
            ds_full = os.path.join(self.ds_path, 'AlgoClientOutput', self.cfg["misc_global"]["detection_mode"])
            msg += ', data_path(size {}): {}'.format(self.out_ds_size, ds_full)
        return msg


class VisAgent(Agent):
    def __init__(self, cfg: json, name: str, sock: socket = None,
                 client_adr: tuple = (None, None), active: bool = False):
        super().__init__(cfg, name, sock, client_adr, active)
        return

    def receive_fake_data(self, t: int, expected_msg: int) -> (str, int):
        j_in = {}
        if expected_msg == 'work':
            j_in = {'name': self.name, 'msg': 'done'}
        elif expected_msg == 'post_ctor_client':
            j_in = {'name': self.name, 'msg': 'post_ctor_client', 'extra': {}, "t": "shake msg 5/5"}
        data_in = utils.json_to_string(j_in)
        return data_in, len(data_in)

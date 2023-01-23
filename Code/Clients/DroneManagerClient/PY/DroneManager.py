import math
import random
import time
import numpy as np
from djitellopy import tello
import wizzi_utils as wu
import pyttsx3

global img
FW, BW, R, L, U, D, RR, RL = 'forward', 'backward', 'right', 'left', 'up', 'down', 'rotate right', 'rotate left'
IN_POINT_RADIUS = 30
IN_TRAJECTORY_RADIUS = 20
ATOMIC_CM_DIST = 20
ATOMIC_POWER = 30
SECS_SLEEP_DURING_ATOMIC = 0.5
SECS_SLEEP_POST_ATOMIC = 0.5
DIM_TO_COL = {L: 0, FW: 1, U: 2, R: 0, BW: 1, D: 2}
IS_ENGINE_OPS = False
POS, NEG = 1, -1
IS_DEBUG = True
fig, ax, iterative_scatter, fig_canvas, position_sc = None, None, None, None, None
PLOT_LIMITS_SIDE_LEN = 300
DRONE_SCALE_FACTOR = 35


class Op:
    def __init__(self, op_type: str, num: float):
        self.op_type = op_type
        self.num = num
        return

    def __str__(self):
        string = '{} {}'.format(self.op_type, self.num)
        return string


class DroneManager:
    """
    This class is capable of drone flight management. It connects a TelloDGI via wifi, send commands, and given on-line
    position and orientation of the drone in real time, it can fix the drone back to main route when it deviates from
    it.
    """

    def __init__(self):
        # init fields
        self.current_p, self.current_o, self.drone_ref_direction = None, np.array([-1, 0, 0]), np.array([0, 1, 0])
        self.start_p, self.target_p, self.start_to_target, self.start_target_dist = None, None, None, None
        self.start_to_target_normalized = None
        self.FW_axis, self.L_axis, self.U_axis = None, None, None
        self.vec_to_route_FW_size, self.vec_to_route_L_size, self.vec_to_route_U_size = None, None, None
        self.sign_FW, self.sign_L, self.sign_U = None, None, None
        self.drone, self.is_drone_detected = None, False
        self.is_finished_ops, self.current_op = False, None
        self.is_drone_detected, self.fig_iter = True, 0
        self.positions = []
        self.speech_engine = pyttsx3.init()
        self.init_time = time.time()
        self.connect_drone()
        return

    def connect_drone(self) -> None:
        """
        This function connects to the TelloDGI drone via WIFI
        @return:
        """
        self.drone = tello.Tello()
        self.drone.connect()
        print(self.drone.get_battery())
        self.drone.streamon()

    def takeoff(self) -> None:
        """
        This functions send to the drone takeoff command
        @return: -
        """
        self.drone.takeoff()
        time.sleep(1)
        return

    def land(self) -> None:
        """
        This functions send to the drone landing command
        @return: -
        """
        self.drone.land()
        return

    def move_forward_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement forward a drone can do
        @return: -
        """
        print('time: ', time.time() - self.init_time)
        print('in move_forward_atomic')
        if self.is_drone_detected:
            if IS_ENGINE_OPS:
                self.drone.send_rc_control(0, ATOMIC_POWER, 0, 0)
                time.sleep(SECS_SLEEP_DURING_ATOMIC)
                self.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
            else:
                self.drone.move_forward(ATOMIC_CM_DIST)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
        else:
            self.drone.send_rc_control(0, 0, 0, 0)
        return

    def move_backward_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement backward a drone can do
        @return: -
        """
        print('time: ', time.time() - self.init_time)
        print('in move_backward_atomic')
        if self.is_drone_detected:
            if IS_ENGINE_OPS:
                self.drone.send_rc_control(0, -ATOMIC_POWER, 0, 0)
                time.sleep(SECS_SLEEP_DURING_ATOMIC)
                self.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
            else:
                self.drone.move_back(ATOMIC_CM_DIST)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
        else:
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_right_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement right a drone can do
        @return: -
        """
        print('time: ', time.time() - self.init_time)
        print('in move_right_atomic')
        if self.is_drone_detected:
            if IS_ENGINE_OPS:
                self.drone.send_rc_control(ATOMIC_POWER, 0, 0, 0)
                time.sleep(SECS_SLEEP_DURING_ATOMIC)
                self.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
            else:
                self.drone.move_right(ATOMIC_CM_DIST)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
        else:
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_left_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement left a drone can do
        @return: -
        """
        print('time: ', time.time() - self.init_time)
        print('in move_left_atomic')
        if self.is_drone_detected:
            if IS_ENGINE_OPS:
                self.drone.send_rc_control(-ATOMIC_POWER, 0, 0, 0)
                time.sleep(SECS_SLEEP_DURING_ATOMIC)
                self.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
            else:
                self.drone.move_left(ATOMIC_CM_DIST)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
        else:
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_up_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement up a drone can do
        @return: -
        """
        print('time: ', time.time() - self.init_time)
        print('in move_up_atomic')
        if self.is_drone_detected:
            if IS_ENGINE_OPS:
                self.drone.send_rc_control(0, 0, ATOMIC_POWER, 0)
                time.sleep(SECS_SLEEP_DURING_ATOMIC)
                self.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
            else:
                self.drone.move_up(ATOMIC_CM_DIST)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
        else:
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_down_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement down a drone can do
        @return: -
        """
        print('time: ', time.time() - self.init_time)
        print('in move_down_atomic')
        if self.is_drone_detected:
            if IS_ENGINE_OPS:
                self.drone.send_rc_control(0, 0, -ATOMIC_POWER, 0)
                time.sleep(SECS_SLEEP_DURING_ATOMIC)
                self.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
            else:
                self.drone.move_down(ATOMIC_CM_DIST)
                time.sleep(SECS_SLEEP_POST_ATOMIC)
        else:
            self.drone.send_rc_control(0, 0, 0, 0)

    def check_if_drone_in_point_sphere(self, center: np.array) -> bool:
        """
        This function checks if the drone is inside a sphere that is centered by given center
        @param center: np.array
        @return:
        """
        print('checks if drone in ', [np.round(center[0], 3), np.round(center[1], 3), np.round(center[2], 3)])
        diff = (center[0:2] - self.current_p[0:2])
        print('diff: ', diff)
        dist = np.linalg.norm(diff, ord=2)
        print('dist from target: ', dist)
        if dist < IN_POINT_RADIUS:
            print('drone arrive to target')
            return True
        return False

    def check_if_drone_in_point_sphere_with_directions(self, threshold: float) -> bool:
        """
        This functions computes if the is any movment in one of the axes that are larger then a given threshold.
        @param threshold: value to measure for movement
        @return: if we need to move or we arrived to destination
        """
        return self.vec_to_route_FW_size < threshold and \
               self.vec_to_route_L_size < threshold
               # self.vec_to_route_U_size < threshold and \

    def compute_drone_directions_for_correction(self) -> None:
        """
        This function computes the smallest segment between the current drone position and the main route, and its
        3 ingredients, one for each axis
        @return: -
        """
        vec_to_route = self.target_p - self.current_p
        vec_to_route_FW = np.multiply(vec_to_route, self.FW_axis)
        vec_to_route_L = np.multiply(vec_to_route, self.L_axis)
        vec_to_route_U = np.multiply(vec_to_route, self.U_axis)
        vectors = [vec_to_route_FW, vec_to_route_L, vec_to_route_U]
        cols = DIM_TO_COL.values()
        for vec in vectors:
            for col in cols:
                if -0.003 <= vec[col] <= 0.003:
                    vec[col] = 0

        self.vec_to_route_FW_size = np.linalg.norm(vec_to_route_FW, ord=2)
        self.vec_to_route_L_size = np.linalg.norm(vec_to_route_L, ord=2)
        self.vec_to_route_U_size = np.linalg.norm(vec_to_route_U, ord=2)

        self.sign_FW = POS if np.dot(vec_to_route, self.FW_axis) >= 0 else NEG
        self.sign_L = POS if np.dot(vec_to_route, self.L_axis) >= 0 else NEG
        self.sign_U = POS if np.dot(vec_to_route, self.U_axis) >= 0 else NEG
        if IS_DEBUG:
            print('current_p: ', self.current_p)
            print('vec_to_route: ', vec_to_route)
            print('vec_to_route_FW: ', vec_to_route_FW)
            print('vec_to_route_L: ', vec_to_route_L)
            print('vec_to_route_U: ', vec_to_route_U)
            print('vec_to_route_FW_size: ', self.vec_to_route_FW_size)
            print('vec_to_route_L_size: ', self.vec_to_route_L_size)
            print('vec_to_route_U_size: ', self.vec_to_route_U_size)
            print('sign_FW: ', self.sign_FW)
            print('sign_L: ', self.sign_L)
            print('sign_U: ', self.sign_U)

    def move_drone_in_correction_directions(self):
        """
        This function assumes direction and distances back to main route are already been computed, and move the drone
        in these directions.
        @return: -
        """
        if self.vec_to_route_FW_size > ATOMIC_CM_DIST / 2:
            if self.sign_FW == POS:
                self.move_forward_atomic()
            elif self.sign_FW == NEG:
                self.move_backward_atomic()
        if self.vec_to_route_L_size > ATOMIC_CM_DIST / 2:
            if self.sign_L == POS:
                self.move_left_atomic()
            elif self.sign_L == NEG:
                self.move_right_atomic()
        if False and self.vec_to_route_U_size > ATOMIC_CM_DIST / 2:
            if self.sign_U == POS:
                self.move_up_atomic()
            elif self.sign_U == NEG:
                self.move_down_atomic()

    def get_required_o(self, op_type: str) -> np.array:
        """
        This function returns the required orientation, depends on the op_type, assuming that the 3 axes are already
        being computed before.
        @param op_type: str
        @return:
        """
        required_o = None
        print(op_type == FW)
        if op_type == FW:
            required_o = self.FW_axis
        elif op_type == BW:
            required_o = -self.FW_axis
        elif op_type == R:
            required_o = -self.L_axis
        elif op_type == L:
            required_o = self.L_axis
        elif op_type == U:
            required_o = self.U_axis
        elif op_type == D:
            required_o = -self.U_axis
        for col in DIM_TO_COL.values():
            if -0.003 <= required_o[col] <= 0.003:
                required_o[col] = 0
        return required_o

    def init_start_and_target_p(self, command: Op) -> None:
        """
        This function initialize the start and target points for a given command, other measurements s.a.
        distances are being measured as well.
        @param command: Op
        @return: -
        """
        if self.target_p is None:
            self.start_p = self.current_p
        else:
            self.start_p = self.target_p
        op_type = command.op_type
        cm = command.num
        required_o = self.get_required_o(op_type)
        self.target_p = self.start_p + (required_o * cm)
        print('required_o: ', required_o)
        print('new_target_p: ', self.target_p)
        self.start_to_target = self.target_p - self.start_p
        self.start_to_target_normalized = self.start_to_target / np.linalg.norm(self.start_to_target, ord=2)
        self.start_target_dist = np.linalg.norm(self.start_to_target, ord=2)

    def compute_axes(self) -> None:
        """
        This dunction computes the 3 axes - forward, left and up.
        @return: -
        """
        # FW_axis = self.current_o
        # FW_axis = [1,0.75,2]
        FW_axis = [0.41095832, 0.27051443, 0.87059474]
        angle = -np.pi / 2.0
        rotation_mat = np.array([[np.cos(angle), -1 * np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
        current_o_2d = self.current_o[0:2]
        required_o_2d = rotation_mat @ current_o_2d
        # L_axis = np.array([required_o_2d[0], required_o_2d[1], 0])
        L_axis = np.array([0.90300192, 0.02076294,  -0.42913452])
        U_axis = np.cross(FW_axis, L_axis)
        self.FW_axis = FW_axis / np.linalg.norm(FW_axis, ord=2)
        self.L_axis = L_axis / np.linalg.norm(L_axis, ord=2)
        self.U_axis = U_axis / np.linalg.norm(U_axis, ord=2)
        vectors = [self.FW_axis, self.L_axis, self.U_axis]
        cols = DIM_TO_COL.values()
        for vec in vectors:
            for col in cols:
                if -0.003 <= vec[col] <= 0.003:
                    vec[col] = 0

        if IS_DEBUG:
            print('FW_axis: ', self.FW_axis)
            print('L_axis: ', self.L_axis)
            print('U_axis: ', self.U_axis)

    def translate_drone(self, command: Op) -> None:
        """
        This function capable of handling transition operation (forward/backward/right/left/up/down).
        @param command: Op
        @return: -
        """
        if command.op_type in [U, D]:
            if command.op_type == U:
                self.move_up_atomic()
            elif command.op_type == D:
                self.move_down_atomic()
        else:
            self.compute_axes()
            self.init_start_and_target_p(command)
            self.positions.append(self.target_p)
            while not self.check_if_drone_in_point_sphere(self.target_p):
                if not self.is_drone_detected:
                    print('drone was not detected')
                    time.sleep(0.1)
                    continue
                self.positions.append(self.current_p)
                # current_drone = generate_drone_3d_points(self.current_p, self.current_o, self.drone_ref_direction)
                print('positions: ', self.positions)
                # plot_position(current_drone, np.array(self.positions, dtype=float), self.fig_iter)
                print('current_p: ', self.current_p, ', current_o: ', self.current_o)
                if self.current_p is None or self.current_o is None:
                    continue
                self.compute_axes()
                self.compute_drone_directions_for_correction()
                if self.check_if_drone_in_point_sphere_with_directions(ATOMIC_CM_DIST):
                    print('drone does not need to move in any direction anymore')
                    break
                print('translate_drone iteration number ', self.fig_iter)
                self.move_drone_in_correction_directions()
                print('***************************')
                self.fig_iter += 1
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
            print('finished translate operation: ', command.op_type)

    def rotate_drone(self, command: Op) -> None:
        """
        This function handle a rotation-in-place of the drone, given a number of degrees to rotate the drone to.
        @param command: Op
        @return: -
        """
        pass

    def set_P_O(self, current_p: np.array, current_o: np.array) -> None:
        """
        current position and orientation setter
        @param current_p: np.array
        @param current_o: np.array
        @return: -
        """
        self.current_p = current_p
        self.current_o = current_o

    def handle_commands(self, commands: list) -> None:
        """
        This function gets a list of commands to the drone, and run the command one after the other.
        @param commands: list
        @return: -
        """
        time.sleep(10)
        self.takeoff()
        time.sleep(10)
        while self.current_p is None or self.current_o is None:
            time.sleep(0.5)
        for command in commands:
            print('current_op: ', command.op_type)
            self.current_op = command.op_type
            if command.op_type in [FW, BW, R, L, U, D]:
                self.translate_drone(command)
            elif command.op_type in [RR, RL]:
                self.rotate_drone(command)
            # txt_to_speak = str('finished ' + command.op_type + ' command')
            # self.speech_engine.say(txt_to_speak)
            # self.speech_engine.runAndWait()
        self.land()
        self.is_finished_ops = True
        print('finished all commands successfully')


def rotate_lines(lines_x: list, lines_y: list) -> (list, list):
    """
    This functions rotates randomly the drone propelors
    @param lines_x: x values of the lines
    @param lines_y: y values of the lines
    @return:
    """
    lines_x_final, lines_y_final = [], []
    for line_set_i in range(len(lines_x)):
        angle = np.random.rand() * np.pi
        rotation_mat = np.array([[np.cos(angle), -1 * np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
        line_x = lines_x[line_set_i]
        line_y = lines_y[line_set_i]
        line_x_final, line_y_final = [], []
        for i in range(len(line_x)):
            x = line_x[i]
            y = line_y[i]
            p = np.array([x, y])
            p_rot = rotation_mat @ p
            line_x_final.append(p_rot[0])
            line_y_final.append(p_rot[1])
        lines_x_final.append(line_x_final)
        lines_y_final.append(line_y_final)
    return lines_x_final, lines_y_final


def generate_drone_3d_points(current_p: np.array, current_o: np.array, drone_ref_direction: np.array) -> list:
    """
    This function computes a point cloud of a drone, and returns it, rotated as desired in a 3d point cloud
    @param current_p: current position
    @param current_o: current orientation
    @param drone_ref_direction: the initial direction the drone is rotated according to
    @return: the drone 3d point cloud
    """
    number_of_propellers = 4
    circles_x, circles_y = [], []

    for _ in range(number_of_propellers):
        circle_x, circle_y = [], []
        for _ in range(0, 1000):
            angle = random.uniform(0, 1) * (math.pi * 2)
            circle_x.append(math.cos(angle))
            circle_y.append(math.sin(angle))
        circles_x.append(circle_x)
        circles_y.append(circle_y)

    proj = np.asarray([1, 1]) / np.linalg.norm(np.asarray([1, 1]), ord=2)
    proj_b = -proj
    lines_x, lines_y = [], []
    for i in range(number_of_propellers):
        values = np.linspace(proj_b[0], proj[0], 1000)
        line_x = list(values)
        line_y = list(values)
        lines_x.append(line_x)
        lines_y.append(line_y)

    lines_x, lines_y = rotate_lines(lines_x, lines_y)

    center_val = 3
    centers = np.array([[center_val, center_val],
                        [-center_val, center_val],
                        [-center_val, -center_val],
                        [center_val, -center_val]])
    for center_i in range(number_of_propellers):
        for i in range(len(circles_x[center_i])):
            circles_x[center_i][i] += centers[center_i][0]
            circles_y[center_i][i] += centers[center_i][1]
        for i in range(len(lines_x[center_i])):
            lines_x[center_i][i] += centers[center_i][0]
            lines_y[center_i][i] += centers[center_i][1]

    # two main cross lines
    values = np.linspace(centers[2][0], centers[0][0], 1000)
    line_x = list(values)
    line_y = list(values)
    lines_x.append(line_x)
    lines_y.append(line_y)
    values = np.linspace(centers[1][0], centers[3][0], 1000)
    line_x = list(values)
    line_y = list(centers[3][1] - values + centers[1][1])
    lines_x.append(line_x)
    lines_y.append(line_y)

    # add orientation arrow
    ARROW_MAIN_LINE_SIZE = 1000
    ARROW_LEN = 5
    ARROW_MAIN_LINE_TIPS_SIZE = 300
    arrow_main_line = np.array([[(centers[0][0] + centers[1][0]) / 2.0, 0]] * ARROW_MAIN_LINE_SIZE)
    for i in range(len(arrow_main_line) - 1):
        arrow_main_line[i + 1][1] += arrow_main_line[0][1] + ARROW_LEN * i / len(arrow_main_line)

    arrow_main_line_start, arrow_main_line_end = arrow_main_line[0], arrow_main_line[-1]
    arrow_main_line_fraction = (arrow_main_line_start + arrow_main_line_end) * 8 / 10
    arrow_tip_vec = arrow_main_line_fraction - arrow_main_line_end
    angle = np.pi / 4
    rotation_mat = np.array([[np.cos(angle), -1 * np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
    arrow_tip_vec_right = rotation_mat @ arrow_tip_vec + arrow_main_line_end
    angle = -np.pi / 4
    rotation_mat = np.array([[np.cos(angle), -1 * np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
    arrow_tip_vec_left = rotation_mat @ arrow_tip_vec + arrow_main_line_end
    arrow_tips = []
    for vec in [arrow_tip_vec_right, arrow_tip_vec_left]:
        for i in range(ARROW_MAIN_LINE_TIPS_SIZE):
            p = arrow_main_line_end - (arrow_main_line_end - vec) * (i + 1) / ARROW_MAIN_LINE_TIPS_SIZE
            arrow_tips.append(p)
    arrow_final = []
    arrow_final.extend(arrow_main_line)
    arrow_final.extend(arrow_tips)

    # get all points
    all_points_x, all_points_y = [], []
    for circle_x_i in range(len(circles_x)):
        for i in range(len(circles_x[circle_x_i])):
            all_points_x.append(circles_x[circle_x_i][i])
            all_points_y.append(circles_y[circle_x_i][i])
    for line_x_i in range(len(lines_x)):
        for i in range(len(lines_x[line_x_i])):
            all_points_x.append(lines_x[line_x_i][i])
            all_points_y.append(lines_y[line_x_i][i])
    for p_i in range(len(arrow_final)):
        all_points_x.append(arrow_final[p_i][0])
        all_points_y.append(arrow_final[p_i][1])

    # rotate the points and add them to a single list, we assume that the drone is rotated from [0, 1, 0]
    dot = np.dot(current_o, drone_ref_direction)
    print('current_o: ', current_o)
    print('drone_ref_direction: ', drone_ref_direction)
    print('dot: ', dot)
    angle = -np.arccos(round(dot, 4))
    rotation_mat = np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    all_points = []
    for i in range(len(all_points_x)):
        p = np.array([all_points_x[i], all_points_y[i]])
        p = rotation_mat.T @ p
        all_points.append([p[0] * DRONE_SCALE_FACTOR + current_p[DIM_TO_COL[L]],
                           p[1] * DRONE_SCALE_FACTOR + current_p[DIM_TO_COL[FW]],
                           current_p[DIM_TO_COL[U]]])

    # all_points_x, all_points_y = [], []
    # for p in all_points:
    #     all_points_x.append(p[0])
    #     all_points_y.append(p[1])
    # plt.scatter(all_points_x, all_points_y)
    # plt.show()
    return all_points


def plot_position(current_drone: np.array, positions: np.array, iteration: int) -> None:
    """
    This function plots the 3d scene with a drone, the paths it took so far, and the trajectory it made so far
    @param current_drone: the drone cloud points
    @param positions: the trajectory so far
    @param iteration: for the title
    @return:
    """
    global fig
    global ax
    global iterative_scatter
    global fig_canvas
    global position_sc
    if iteration == 0:
        fig, ax, iterative_scatter, fig_canvas = wu.pyplt.plot_3d_iterative_figure(
            scatter_dict={'c': 'b', 'marker_size': 0.01, 'marker': '.', 'label': 'Drone '},
            main_title='3d scatter plot',
            resize=1.5,
            plot_location='top_center',
            x_y_z_lims=[-PLOT_LIMITS_SIDE_LEN, PLOT_LIMITS_SIDE_LEN, -PLOT_LIMITS_SIDE_LEN, PLOT_LIMITS_SIDE_LEN,
                        -PLOT_LIMITS_SIDE_LEN, PLOT_LIMITS_SIDE_LEN],
            fig_face_color=None,
            ax_background=None,
            ax_labels_and_ticks_c=None,
            add_center={'c': 'orange', 'marker': 'x', 'marker_size': 150, 'label': 'Center'},
            zoomed=False,
            view={'azim': 89.0, 'elev': -118.0},
            render_d={'block': False, 'pause': 0.0001}
        )

        # # CUSTOM ADD ON 1 - update each round
        position_sc_dict = {'c': 'r', 'marker_size': 5, 'marker': 'o', 'label': 'Traj'}
        position_sc = ax.scatter(
            0, 0, 0,
            c=position_sc_dict['c'], marker=position_sc_dict['marker'],
            s=position_sc_dict['marker_size'], label=position_sc_dict['label']
        )
    else:
        # print('positions: ', positions)
        # num_points = len(positions)
        # colors = get_random_color_map(num_points)
        # misc_tools.sleep(seconds=0)

        wu.pyplt.update_3d_scatters(
            scatter=iterative_scatter,
            fig_canvas=fig_canvas,
            data=np.array(current_drone),
            colors=None,
            new_title='iter {}'.format(iteration),
            render_d=None
        )

        position_sc._offsets3d = positions.T

        # if colors_sc is not None:
        #     position_sc._facecolor3d = colors
        #     position_sc._edgecolor3d = colors

        wu.pyplt.render(block=False)
    return

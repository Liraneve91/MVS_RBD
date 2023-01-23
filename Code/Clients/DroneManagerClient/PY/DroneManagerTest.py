import threading

from Clients.DroneManagerClient.PY.DroneManager import *

SIMULATE_SECS_OPERATION = 0.02
OP_LEN = 500
ROT_LEN = np.pi / 4
MU, SIGMA = 0, 0.1


class DroneManagerTest(DroneManager):
    """
    This class is capable of drone flight management. It connects a TelloDGI via wifi, send commands, and given on-line
    position and orientation of the drone in real time, it can fix the drone back to main route when it deviates from
    it.
    """

    def __init__(self):
        super().__init__()
        self.drone_ref_direction = np.array([0, 1, 0])
        self.positions, self.simulation_iteration = [], 0
        self.current_p, self.current_o = np.array([3, 5, 250], dtype=float), np.array([-1, -1, 0],
                                                                                      dtype=float) / np.linalg.norm(
            np.array([-1, -1, 0], dtype=float), ord=2)
        self.start_p, self.target_p, self.start_to_target, self.start_target_dist = None, None, None, None
        self.FW_axis, self.L_axis, self.U_axis = None, None, None
        self.vec_to_route_FW_size, self.vec_to_route_L_size, self.vec_to_route_U_size = None, None, None
        self.sign_FW, self.sign_L, self.sign_U = None, None, None
        self.is_finished_ops = False

    def connect_drone(self):
        pass

    def takeoff(self) -> None:
        """
        This functions send to the drone takeoff command
        @return: -
        """
        time.sleep(SIMULATE_SECS_OPERATION)

    def land(self) -> None:
        """
        This functions send to the drone landing command
        @return: -
        """
        time.sleep(SIMULATE_SECS_OPERATION)

    def move_forward_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement forward a drone can do
        @return: -
        """
        print('in move_forward_atomic')
        self.current_p += ATOMIC_CM_DIST * self.FW_axis
        time.sleep(SIMULATE_SECS_OPERATION)

    def move_backward_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement backward a drone can do
        @return: -
        """
        print('in move_backward_atomic')
        self.current_p -= ATOMIC_CM_DIST * self.FW_axis
        time.sleep(SIMULATE_SECS_OPERATION)

    def move_right_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement right a drone can do
        @return: -
        """
        print('in move_right_atomic')
        self.current_p -= ATOMIC_CM_DIST * self.L_axis
        time.sleep(SIMULATE_SECS_OPERATION)

    def move_left_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement left a drone can do
        @return: -
        """
        print('in move_left_atomic')
        self.current_p += ATOMIC_CM_DIST * self.L_axis
        time.sleep(SIMULATE_SECS_OPERATION)

    def move_up_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement up a drone can do
        @return: -
        """
        print('in move_up_atomic')
        self.current_p += ATOMIC_CM_DIST * self.U_axis
        time.sleep(SIMULATE_SECS_OPERATION)

    def move_down_atomic(self) -> None:
        """
        This function perform the smallest (atomic) movement down a drone can do
        @return: -
        """
        print('in move_down_atomic')
        self.current_p -= ATOMIC_CM_DIST * self.U_axis
        time.sleep(SIMULATE_SECS_OPERATION)

    def rotate_drone(self, command: Op) -> None:
        """
        This function handle a rotation-in-place of the drone, given a number of degrees to rotate the drone to.
        @param command: Op
        @return: -
        """
        angle = command.num
        if command.op_type == RL:
            angle = - angle
        rotation_mat = np.array([[np.cos(angle), -1 * np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
        current_o_2d_rotated = rotation_mat @ self.current_o[0:2]
        current_o = np.array([current_o_2d_rotated[0], current_o_2d_rotated[1], 0])
        current_o = current_o / np.linalg.norm(current_o, ord=2)
        self.current_o = current_o

    def add_noise_and_plot(self, iteration: int) -> None:
        """
        This function add a normal distributed noise tho the drone position and plots its 3d
        @param iteration: int - for figure title
        @return:
        """
        rand_p = np.random.normal(MU, SIGMA, 3) * 100
        rand_o = np.random.normal(MU, SIGMA, 3) * 0.2
        print('rand: ', rand_p)
        current_p = self.current_p + rand_p
        current_o = self.current_o + rand_o
        print('current_p:', current_p)
        print('current_o:', self.current_o)
        current_drone = generate_drone_3d_points(current_p, self.current_o, self.drone_ref_direction)
        self.positions.append(current_p)
        self.set_P_O(current_p, current_o)
        plot_position(current_drone, np.array(self.positions, dtype=float), iteration)

    def translate_drone(self, command: Op) -> None:
        """
        This function capable of handling transition operation (forward/backward/right/left/up/down).
        @param command: Op
        @return: -
        """
        self.compute_axes()
        self.init_start_and_target_p(command)
        self.positions.append(self.target_p)
        while not self.check_if_drone_in_point_sphere(self.target_p):
            self.add_noise_and_plot(self.simulation_iteration)
            print('current_p: ', self.current_p, ', current_o: ', self.current_o)
            if self.current_p is None or self.current_o is None:
                continue
            self.compute_axes()
            self.compute_drone_directions_for_correction()
            self.move_drone_in_correction_directions()
            print('translate_drone iteration number ', self.simulation_iteration)
            print('***************************')
            self.simulation_iteration += 1
        print('finished translate operation: ', command.op_type)

    def handle_commands(self, commands: list) -> None:
        """
        This function gets a list of commands to the drone, and run the command one after the other.
        @param commands: list
        @return: -
        """
        while self.current_p is None or self.current_o is None:
            time.sleep(0.5)
        for command in commands:
            print('current_op: ', command.op_type)
            self.current_op = command.op_type
            if command.op_type in [FW, BW, R, L, U, D]:
                self.translate_drone(command)
            elif command.op_type in [RR, RL]:
                self.rotate_drone(command)
        self.is_finished_ops = True
        print('finished all commands successfully')
        wu.pyplt.render(block=True)


def main() -> None:
    dm = DroneManagerTest()
    commands = [
        Op(FW, OP_LEN),
        Op(R, OP_LEN),
        Op(BW, OP_LEN),
        Op(L, OP_LEN),
        Op(U, OP_LEN),
        Op(RR, ROT_LEN),
        Op(FW, OP_LEN / 2),
        Op(R, OP_LEN / 2),
        Op(BW, OP_LEN / 2),
        Op(L, OP_LEN / 2),
    ]
    print(wu.to_str(commands[0], 'commands[0]'))

    t = threading.Thread(target=dm.handle_commands, args=(commands,))
    t.start()
    # positions = []
    # i = 0
    #
    # while True:
    # rand = np.random.normal(MU, SIGMA, 3) * 100
    # print('rand: ', rand)
    # current_p = dm.current_p + rand
    # print('current_p:', current_p)
    # print('current_o:', dm.current_o)
    # current_drone = generate_drone_3d_points(current_p, dm.current_o)
    # positions.append(current_p)
    # dm.set_P_O(current_p, dm.current_o)
    # plot_position(current_drone, np.array(positions, dtype=float), i)
    # # time.sleep(0.05)
    # i += 1
    # print('position len: ', len(positions))
    # print('finished round ', i)
    # print('*********************************************************')
    # if dm.is_finished_ops:
    #     break
    # wu.pyplt.render(block=True)


if __name__ == '__main__':
    main()

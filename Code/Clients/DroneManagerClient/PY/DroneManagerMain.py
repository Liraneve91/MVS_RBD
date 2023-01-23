import SharedCode.PY.utilsScript as utils
from Clients.DroneManagerClient.PY.DroneManagerHandler import  DroneManagerHandler


def main():
    drone_handler, max_iters = None, 2 ** 64
    try:
        drone_handler = DroneManagerHandler()
        drone_handler.initialize()
        if drone_handler.local_mode:
            max_iters = drone_handler.cfg['misc_out']['local_mode_cfg']['max_iters']
    except (ValueError, ConnectionResetError, ConnectionAbortedError) as err:
        print(err)
        exit(-100)  # no point to continue

    times_ti = []
    try:
        for t in range(max_iters):
            begin_ti = utils.get_time_now()
            print('\nt={}:'.format(t))
            drone_handler.do_work(iteration_t=t, expected_msg='work')
            times_ti.append(utils.get_time_now() - begin_ti)
            print('DONE round={}: total time={}'.format(t, utils.get_time_str_seconds(begin_ti)))
    except (ValueError, ConnectionResetError, ConnectionAbortedError) as err:
        print(err)

    utils.rounds_summary(times_ti)
    return


if __name__ == '__main__':
    main()

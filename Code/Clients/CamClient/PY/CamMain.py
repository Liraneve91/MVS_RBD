import sys

sys.path.append('../../../')  # adds root as MVS/Code2 - for running in shell
import SharedCode.PY.utilsScript as utils  # noqa: E402
from Clients.CamClient.PY.CamHandler import CamHandler  # noqa: E402


def main():
    cam, expected_work_msg, max_iters = None, -1, 2 ** 64
    try:
        cam = CamHandler()
        cam.initialize()
        if cam.local_mode:
            max_iters = cam.cfg['misc_cam']['local_mode_cfg']['max_iters']
            if cam.cfg['misc_global']['system_mode'] == 'mapping':
                expected_work_msg = 'work'
            elif cam.cfg['misc_global']['system_mode'] == 'calibration':
                expected_work_msg = 'calib'
    except (ValueError, ConnectionResetError, ConnectionAbortedError) as err:
        print(err)
        exit(-100)  # no point to continue

    times_ti = []
    # try:
    for t in range(max_iters):
        begin_ti = utils.get_time_now()
        print('\nt={}:'.format(t))
        cam.do_work(iteration_t=t, expected_msg=expected_work_msg)
        times_ti.append(utils.get_time_now() - begin_ti)
        print('DONE round={}: total time={}'.format(t, utils.get_time_str_seconds(begin_ti)))
    # except (ValueError, ConnectionResetError, ConnectionAbortedError) as err:
    #     print(err)

    utils.rounds_summary(times_ti)
    return


if __name__ == '__main__':
    # pr = utils.start_profiler()
    main()
    # print(utils.end_profiler(pr, rows=15))

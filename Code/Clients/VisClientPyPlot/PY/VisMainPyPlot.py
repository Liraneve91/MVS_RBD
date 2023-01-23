import SharedCode.PY.utilsScript as utils
from Clients.VisClientPyPlot.PY.VisHandlerPyPlot import VisHandlerPyPlot as VisHandler


def main():
    vis, max_iters = None, 2 ** 64
    try:
        vis = VisHandler()
        vis.initialize()
        if vis.local_mode:
            max_iters = vis.cfg['misc_vis']['local_mode_cfg']['max_iters']
    except (ValueError, ConnectionResetError, ConnectionAbortedError) as err:
        print(err)
        exit(-100)  # no point to continue

    times_ti = []
    try:
        for t in range(max_iters):
            begin_ti = utils.get_time_now()
            print('\nt={}:'.format(t))
            vis.do_work(iteration_t=t, expected_msg='work')
            times_ti.append(utils.get_time_now() - begin_ti)
            print('DONE round={}: total time={}'.format(t, utils.get_time_str_seconds(begin_ti)))
    except (ValueError, ConnectionResetError, ConnectionAbortedError) as err:
        print(err)

    utils.rounds_summary(times_ti)
    return


if __name__ == '__main__':
    # pr = utils.start_profiler()
    main()
    # print(utils.end_profiler(pr, rows=15))

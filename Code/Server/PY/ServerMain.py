import Server.PY.ServerHandler as serverApi
import SharedCode.PY.utilsScript as utils


def main():
    max_iters = 2 ** 64
    # max_iters = 21
    server = None
    try:
        server = serverApi.ServerHandler()
    except (ValueError, ConnectionResetError, ConnectionAbortedError) as err:
        print(err)
        exit(-100)  # no point to continue

    t_start, t_interval = 0, float("inf")  # normal 0 to inf, use every t
    if server.cfg['misc_server']['custom_t']['ues_custom_t']:
        t_start = server.cfg['misc_server']['custom_t']['t_start']
        t_end = server.cfg['misc_server']['custom_t']['t_end']
        t_interval = t_end - t_start + 1

    times_ti = []
    try:
        for t in range(max_iters):
            begin_ti = utils.get_time_now()
            t_custom = int(t % t_interval) + t_start
            print('\nt={} (custom_t={}):'.format(t, t_custom))
            server.ask_agents_data(t=t_custom)
            server.analyze_data(t=t_custom)
            server.visualize_object(t=t_custom)
            times_ti.append(utils.get_time_now() - begin_ti)
            # print('DONE round={}: total time={}'.format(t, utils.get_time_str_seconds(begin_ti)))
    except (ValueError, ConnectionResetError, ConnectionAbortedError) as err:
        print(err)

    utils.rounds_summary(times_ti)
    return


if __name__ == '__main__':
    main()

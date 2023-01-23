import datetime
import inspect
import itertools
import json
import math
import os
import pickle
import socket
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

WARNING = 'WARNING: {} from {} ' + 'X' * 100
DIM_3 = 3
DIM_2 = 2

# relative to project home. e.g. Code2/SharedCode/CamClientMeta.json
SHARED_CONSTANTS_PATH = 'SharedCode/GlobalConstants.json'
SERVER_ADDRESS_PATH = 'SharedCode/ServerAddress.json'

RGB_BLACK = (0, 0, 0)
RGB_ORANGE = (255, 165, 0)
BGR_ORANGE = (0, 165, 255)
RGBA_ORANGE = (1.0, 0.64, 0.0, 1.0)
RGBA_GREEN = (0.0, 0.5, 0.0, 1.0)
BGR_GREEN = (0, 255, 0)
BGR_RED = (0, 0, 255)
BGR_YELLOW = (0, 255, 255)


def load_jsons(files_path: list, ack: bool = True) -> dict:
    """ :returns a dict of the jsons concatenated """
    all_in_one_dict = {}
    len_keys_json = 0
    for file_path in files_path:
        j = load_json(file_path, ack=False)
        len_keys_json += len(j)
        all_in_one_dict.update(j)
    if ack:
        print('{} loaded (type {})'.format(files_path, type(all_in_one_dict)))
        print(json_to_string(all_in_one_dict, indent=4))
    assert len_keys_json == len(all_in_one_dict), 'Duplicated keys found: {}'.format(files_path)
    return all_in_one_dict


def load_json(file_path: str, ack: bool = True) -> dict:
    """ :returns a dict """
    ret_dict = {}
    if os.path.exists(file_path):
        ret_dict = json.load(open(file_path))
        if ack:
            print('{} loaded (type {})'.format(file_path, type(ret_dict)))
            print(json_to_string(ret_dict, indent=4))
    else:
        print('file {} doesnt exists'.format(file_path))
    return ret_dict


def save_json(file_path: str, j: dict, indent: int = 2, sort_keys: bool = True, ack: bool = True) -> None:
    json.dump(j, open(file_path, 'w'), indent=indent, sort_keys=sort_keys)
    if ack:
        print('{} saved successfully'.format(file_path))
        print(json_to_string(j, indent=indent, sort_keys=sort_keys))
    return


def json_to_string(j: dict, indent: int = -1, sort_keys: bool = True):
    """
    e.g.
    str = json_to_string(cfg, indent=-1)  # different from 0
    print(str) # normal string for TCP
    print(json_to_string(cfg, indent=4))  # pretty print
    print(json_to_string(cfg, indent=0))  # pretty print - 0 indentations
    """
    if indent == -1:
        indent = None
    return json.dumps(j, indent=indent, sort_keys=sort_keys)


def string_to_json(j_str: str) -> json:
    """
    e.g.
    str = json_to_string(cfg, indent=None)
    j = string_to_json(str)
    """
    return json.loads(j_str)


def start_profiler():
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    return pr


def end_profiler(pr, rows: int = 10) -> str:
    import pstats
    import io
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(rows)
    return s.getvalue()


def get_time_now():
    return time.time()


def get_time_str(start_time) -> str:
    hours, rem = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def get_time_str_seconds(start_time) -> str:
    return '{:.3f} seconds'.format(time.time() - start_time)


def convert_msg_consts(cfg: dict, code_int: int) -> str:
    """
    :param cfg: dict with keys strings and values codes
    e.g. key['CamClient'] == 300
    :param code_int:the code e.g. 300
    :return: the value e.g. 'CamClient'
    """
    string = ''
    for k, v in cfg.items():
        if v == code_int:
            string = k
            break

    # string = list(cfg.keys())[list(cfg.values()).index(code_int)]  # also work
    return string


def open_server(server_address: tuple = ('localhost', 10000), ack: bool = True, tabs: int = 1) -> socket:
    if ack:
        print('{}Opening server on IP,PORT {}'.format(tabs * '\t', server_address))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(server_address)
    sock.listen(1)
    return sock


def get_host_name(ack: bool = True, tabs: int = 1) -> str:
    """ :return: hostname """
    hostname = socket.gethostname()
    if ack:
        print("{}Computer Name: {}".format(tabs * '\t', hostname))
    return hostname


def get_ipv4(ack: bool = True, tabs: int = 1) -> str:
    """ :return ipv4 address of this computer """
    ipv4 = socket.gethostbyname(get_host_name(ack=True, tabs=0))
    if ack:
        print("{}Computer IP Address: {}".format(tabs * '\t', ipv4))
    return ipv4


def send_msg(connection: socket, buflen: int, data: str, msg_end: str) -> None:
    data_e = str.encode(data + msg_end)
    data_e_len = len(data_e)
    for i in range(0, data_e_len, buflen):
        chunk_i = data_e[i:i + buflen]
        connection.send(chunk_i)
    return


def create_client(server_address: tuple = ('localhost', 10000), ack: bool = True, tabs: int = 1) -> socket:
    print("connecting to:")
    print(server_address)
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(server_address)
            return s
        except Exception as e:
            print(e.args[0])
            time.sleep(3)


def receive_msg(connection: socket, buflen: int, msg_end: str) -> (str, int):
    data_in = ''
    saw_end_delimiter = False
    while not saw_end_delimiter:
        data_in += connection.recv(buflen).decode('utf-8')
        if not data_in:
            break  # empty transmission
        if data_in.endswith(msg_end):
            data_in = data_in.replace('$#$#', '')
            saw_end_delimiter = True

    # data_in = connection.recv(buflen).decode('utf-8')
    data_in_len = len(data_in)
    return data_in, data_in_len


def buffer_print(data: str, original_size: int, prefix: str, tabs: int = 1):
    msg = tabs * '\t'
    if original_size < 2000:
        msg += '{}: {} (bytes sent={})'.format(prefix, data, original_size)
    else:  # if really long message - dont print data
        msg += '{}: {} ... message is too long to display (bytes sent={:,})'.format(prefix, data[:2000], original_size)
    print(msg)
    return


def get_fake_data_uniform(low=0, high=10, size=(1, 1), round_digits=3) -> list:
    fake_data_ti = np.random.uniform(low=low, high=high, size=size)
    fake_data_ti = np.round(fake_data_ti, round_digits)
    return fake_data_ti.tolist()


def get_fake_data_uniform_int(low=0, high=10, size=(3, 4), dtype=None):
    """ if dtype returns np array in this type. else list"""
    fake_data_ti = np.random.random_integers(low=low, high=high, size=size)
    if dtype is not None:
        fake_data_ti = np.array(fake_data_ti, dtype=dtype)
    else:
        fake_data_ti = fake_data_ti.tolist()
    return fake_data_ti


def list_1d_to_str(floats: list, digits: int = 4):
    """ |floats| = nx1 """
    p = '{:,.' + str(digits) + 'f}'
    out = '['
    for i, num in enumerate(floats):
        out += p.format(num)
        out += ',' if (i + 1) < len(floats) else ''
    out += ']'
    return out


def rounds_summary(times: list) -> None:
    if len(times) > 0:  # try print time avg before exit
        print('\n')
        print('Rounds Summary:')
        print('\tTotal rounds = {}'.format(len(times)))
        print('\tTotal run time = {:.3f} seconds'.format(np.sum(times)))
        print('\tAvg   run time = {:.3f} seconds (std = {:.3f})'.format(np.mean(times), np.std(times)))
        if np.mean(times) > 0.0001:
            print('\t{:.2f} FPS'.format(1 / np.mean(times)))
    return


def get_mac_address():
    from uuid import getnode as get_mac
    mac = get_mac()
    mac = ':'.join(("%012X" % mac)[i:i + 2] for i in range(0, 12, 2))
    return mac


def get_line_number(depth: int = 1) -> str:
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.lineno)
    except IndexError:
        pass
    return ret_val


def get_function_name(depth: int = 1) -> str:
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.function)
    except IndexError:
        pass
    return ret_val


def get_file_name(depth: int = 1) -> str:
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.filename)
    except IndexError:
        pass
    return ret_val


def get_function_name_and_line(depth: int = 1) -> str:
    # for scope in inspect.stack():  # see all scopes
    #     print(scope)
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{} line {}'.format(scope_1_back.function, scope_1_back.lineno)
    except IndexError:
        pass
    return ret_val


def timer(seconds: int, action: str = '', tabs: int = 1) -> None:
    """ counts till seconds or block"""
    if seconds is None:
        input('{}Press any key for {}...'.format(tabs * '\t', action))
    else:
        time_in_future = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
        print('{}{} IN: {}'.format(tabs * '\t', action, seconds), end='', flush=True)
        while time_in_future > datetime.datetime.now():
            time.sleep(1)
            seconds -= 1
            print(' {}'.format(seconds), end='', flush=True)
        print('')
    return


def display_open_cv_image(img: np.array, ms: int = 0, title: str = '', x_y: tuple = (0, 0),
                          destroy_windows: bool = False):
    if isinstance(img, list):
        img = np.array(img, dtype='uint8')
    if img.dtype != 'uint8':
        np.array(img, dtype='uint8')
    cv2.imshow(title, img)
    if x_y is not None:
        cv2.moveWindow(title, x_y[0], x_y[1])

    cv2.waitKey(ms)
    if destroy_windows:
        cv2.destroyAllWindows()
    return


def resize_opencv_image(img, scale_percent: float):
    width = math.ceil(img.shape[1] * scale_percent)
    height = math.ceil(img.shape[0] * scale_percent)
    dim = (width, height)
    resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_image


def unpack_list_imgs_to_big_image(imgs: list, resize: float, grid: tuple) -> np.array:
    for i in range(len(imgs)):
        if isinstance(imgs[i], list):
            imgs[i] = np.array(imgs[i], dtype='uint8')
        if imgs[i].dtype != 'uint8':
            imgs[i] = np.array(imgs[i], dtype='uint8')
        if resize > 0:
            imgs[i] = resize_opencv_image(imgs[i], resize)
        if len(imgs[i].shape) == 2:  # just x,y: gray scale image
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR)
    imgs_n = len(imgs)
    if imgs_n == 1:
        big_img = imgs[0]
    else:
        padding_bgr = list(BGR_RED)
        height, width, cnls = imgs[0].shape
        rows, cols = grid
        big_img = np.zeros(shape=(height * rows, width * cols, cnls), dtype='uint8') + 255

        row_ind, col_ind = 1, 1
        for i, img in enumerate(imgs):
            h_begin, h_end = height * (row_ind - 1), height * row_ind
            w_begin, w_end = width * (col_ind - 1), width * col_ind
            big_img[h_begin:h_end, w_begin:w_end, :] = img  # 0

            if rows > 1:  # draw bounding box on the edges. no need if there is 1 row or 1 col
                big_img[h_begin, w_begin:w_end, :] = padding_bgr
                big_img[h_end - 1, w_begin:w_end - 1, :] = padding_bgr
            if cols > 1:
                big_img[h_begin:h_end, w_begin, :] = padding_bgr
                big_img[h_begin:h_end, w_end - 1, :] = padding_bgr

            col_ind += 1
            if col_ind > cols:
                col_ind = 1
                row_ind += 1
    return big_img


def display_open_cv_images(
        imgs: list,
        ms: int = 0,
        title: str = '',
        x_y: tuple = (0, 0),
        resize: float = 0.0,
        grid: tuple = (1, 2),
        destroy_windows: bool = False
):
    """
    :param imgs: list of RGB and gray scale images
    :param ms: time to wait
    :param title: main title
    :param x_y: top left corner of the image on relative to the screen
    :param resize: if images are too big, give a smaller size e.g. 60 -> 60% of images size
    :param grid: size of rows and cols of the new image. e.g. (2,1) 2 rows with 1 img on each
    :param destroy_windows: whether to cv2.destroyAllWindows() after time 'ms' passed
    :return:
    """
    imgs_n = len(imgs)
    if imgs_n > 0:
        total_slots = grid[0] * grid[1]
        assert imgs_n <= total_slots, 'grid has {} total_slots, but len(imgs)={}'.format(total_slots, imgs_n)
        big_img = unpack_list_imgs_to_big_image(imgs, resize, grid)
        display_open_cv_image(big_img, ms, title, x_y, destroy_windows)
    return


def display_open_cv_images_for_demo(
        imgs: list,
        ms: int = 0,
        title: str = '',
        x_y: tuple = (0, 0),
        resize: float = 0.0,
        destroy_windows: bool = False
):
    """
    :param imgs: list of RGB and gray scale images
    :param ms: time to wait
    :param title: main title
    :param x_y: top left corner of the image on relative to the screen
    :param resize: if images are too big, give a smaller size e.g. 60 -> 60% of images size
    :param destroy_windows: whether to cv2.destroyAllWindows() after time 'ms' passed
    :return:
    """
    imgs_n = len(imgs)
    mac = title[-14:-12]
    for i in range(imgs_n):
        # big_img = unpack_list_imgs_to_big_image(imgs, resize, grid)
        img = imgs[i]
        img = resize_opencv_image(img, resize)
        current_title = mac + '/' + str(i)
        if current_title in ['8C/0', '3B/1', '57/0', '57/1', '8D/0', '8D/1', 'C6/0', 'C6/1']:
            img = cv2.flip(img, 0)
        if current_title in ['3B/1', '8C/0', '57/0', '57/1', 'C6/0', 'C6/1', '8D/0', '8D/1']:
            img = cv2.flip(img, 1)
        cv2.putText(img, current_title, (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        display_open_cv_image(img, ms, current_title, x_y, destroy_windows)
    return


def get_current_universal_time() -> str:
    diff = datetime.datetime.utcnow() - datetime.datetime(2021, 1, 1, 0, 0, 0)
    timestamp = diff.days * 24 * 60 * 60 + diff.microseconds / 1000
    return str(timestamp)


def get_time_stamp_for_folder_name() -> str:
    date_time_obj = datetime.datetime.now()
    timestamp_str = date_time_obj.strftime("%Y_%m_%d")
    return timestamp_str


def get_current_time_stamp() -> str:
    """
    https://strftime.org/
    timestampStr = dateTimeObj.strftime("%Y_%m_%d_%H_%M_%S_%f") >>> 2020_07_19_13_36_24_597247
    """
    date_time_obj = datetime.datetime.now()
    timestamp_str = date_time_obj.strftime("%Y_%m_%d_%H_%M_%S")
    return timestamp_str


def create_dir_if_not_exists(data_root: str, ack: bool = True, tabs: int = 1):
    if os.path.exists(data_root):
        if ack:
            print('{}{} exists'.format(tabs * '\t', data_root))
    else:
        os.mkdir(data_root)
        if ack:
            print('{}{} created'.format(tabs * '\t', data_root))
    return


def save_img(path: str, img: np.array, ack: bool = True, tabs: int = 1):
    cv2.imwrite(path, img)
    if ack:
        print('{}saving img to {}'.format(tabs * '\t', path))
    return


def load_img(path: str, ack: bool = True, tabs: int = 1):
    img = cv2.imread(path)
    if ack:
        print('{}loaded img from {}'.format(tabs * '\t', path))
    return img


def nCr(n: int, r: int) -> int:
    if r > n:
        return -1
    elif r == n:
        return 1
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def var_to_string(var: np.array, title: str = 'VAR', with_data: bool = False, p: int = 3) -> str:
    """  to string of a np array, list or any object with len """
    if var is None:
        return None
    if isinstance(var, np.ndarray):
        msg = '{}(type={}, shape={}, dtype={})'.format(title, type(var), str(var.shape), var.dtype)
        if with_data:
            msg += ': data={}'.format(np.round(var.tolist(), p).tolist())
    elif isinstance(var, list):
        msg = '{}(size={},type={})'.format(title, len(var), type(var))
        if with_data:
            msg += ': data={}'.format(np.round(var, p).tolist())
    else:
        msg = '{}(type={}, size={})'.format(title, type(var), len(var))
        if with_data:
            msg += ': data={}'.format(var)
    return msg


def nCk(n: int, k: int, as_int: bool = False):
    """
    :param n:
    :param k:
    :param as_int:
    :return: if as_int True: the result of nCk, else the combinations of nCk

    e.g.
    A = np.random.randint(low=-10, high=10, size=(3, 2))
    print('A={}'.format(A.tolist()))

    # let's iterate on every 2 different indices of A
    combs_count = utils.nCk(len(A), k=2, as_int=True)
    print('|A| choose 2={}:'.format(combs_count)) # result is 3

    combs_list = utils.nCk(len(A), k=2)  # result is [[0, 1], [0, 2], [1, 2]]
    for i, comb in enumerate(combs_list):
        print('\tcomb {}={}. A[comb]={}'.format(i, comb, A[comb].tolist()))
    """
    from itertools import combinations
    range_list = np.arange(0, n, 1)
    combs = list(combinations(range_list, k))
    combs = [list(comb) for comb in combs]
    if as_int:
        combs = len(combs)  #
    return combs


def add_arrows_to_axed3d():
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    from matplotlib.patches import FancyArrowPatch

    class Arrow3D(FancyArrowPatch):
        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            try:
                super().draw(renderer)
            except (ValueError, Exception):
                # ValueError wired exceptions from arrows - parallel lines
                pass  # div in zero

    def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
        """Add an 3d arrow to an `Axes3D` instance."""

        arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
        ax.add_artist(arrow)

    setattr(Axes3D, 'arrow3D', _arrow3D)
    return


def add_orientation_arrows_to_points(axes, cameras_params, arrow_color, arrow_len):
    add_arrows_to_axed3d()
    for camera_params in cameras_params:
        p, v = camera_params.p, camera_params.v
        v2 = v * arrow_len  # multiply in desired len
        # print(utils.var_to_string(p, 'p', with_data=True))
        # print(utils.var_to_string(v, 'v', with_data=True))
        # print('xxx', arrow_len, arrow_color, p, v, v2)
        axes.arrow3D(p[0], p[1], p[2],
                     v2[0], v2[1], v2[2],
                     mutation_scale=15,
                     ec='white',
                     fc=arrow_color)
    return


def plot_3d_cube(axes, cube_definition, color='b', label='cube', add_corner_labels=False):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    color_rgba = matplotlib.colors.to_rgba(color, alpha=0.1)
    faces.set_facecolor(color_rgba)
    axes.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    axes.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, color=color, label=label)

    if add_corner_labels:
        for p in points:
            x, y, z = p
            text = str(x) + ', ' + str(y) + ', ' + str(z)
            axes.text(x, y, z, text, zdir=(1, 1, 1))
    return


def update_scatters(scatters: list, datum: list, colors_sets: list, new_title: str, block: bool = False) -> list:
    """
    2d multi plot update
    let N be the number of sub figures
    :param scatters: list of N scatters to update
    :param datum: list of size N of data sets in sizes: (n1x2), (n2x2), ...  # 2 for 2d
    :param colors_sets: list of N of colors in sizes: (n1x3), (n2x3), ... # 4 for RGBA colors. could be empty
    :param new_title: new main title
    :param block: block the plot
    :return:
    """
    plt.suptitle(new_title)
    N = len(scatters)
    for i in range(N):
        sc_i = scatters[i]
        data = datum[i]
        if len(data) > 0:
            sc_i._offsets = datum[i]
            colors = colors_sets[i]
            if len(colors) > 0:
                sc_i._facecolors = colors_sets[i]
                sc_i._edgecolors = colors_sets[i]
    # re-render
    plt.draw()
    plt.show(block=block)
    plt.pause(0.0001)
    return


def plot_horizontal_figures(datum: list, colors_sets: list, titles: list, main_title: str, x_y_lim: list,
                            block: bool = False, plot_location: str = None, resize: int = 0) -> list:
    """
    2d multi plot
    let N be the number of sub figures
    :param datum: x,y sets in size N. x,y is 2d data
    :param colors_sets: list of colors per sub figure. each list in size N or 1. could be empty
    :param titles: list of N titles
    :param main_title:
    :param x_y_lim: plot limits. list of start_x, end_x, start_y, end_y
    :param block: for debug - pause the run
    :param plot_location: whether to move the whole figure
    :param resize: resize the plot tp resize%
    :return:
    """

    N = len(titles)
    # default figsize=(6.4, 4.8)
    figsize = (6.4, 4.8) if resize == 0 else (6.4 * resize, 4.8 * resize)

    fig, axes = plt.subplots(nrows=1, ncols=N, sharex=False, sharey=False, figsize=figsize)
    print(fig.get_size_inches() * fig.dpi)

    if plot_location is not None:
        move_plot(fig, where=plot_location)
    if N == 1:
        axes = [axes]
    fig.suptitle(main_title)
    scatters = []
    for i in range(N):
        ax = axes[i]
        data = datum[i]
        if len(data) > 0:
            xs, ys = data[:, 0], data[:, 1]
            # print(xs.shape, ys.shape, len(colors))
            colors = colors_sets[i]
            if len(colors) > 0:
                sc = ax.scatter(xs, ys, c=colors, marker='.')
            else:
                sc = ax.scatter(xs, ys, c='g', marker='.')
        else:
            sc = ax.scatter([0], [0], c='g', marker='.')
        # ax.grid(False)
        ax.set_title(titles[i])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_y_lim[0], x_y_lim[1])
        ax.set_ylim(x_y_lim[2], x_y_lim[3])
        cx, cy = int((x_y_lim[1] - x_y_lim[0]) / 2), int((x_y_lim[3] - x_y_lim[2]) / 2)
        ax.scatter([cx], [cy], c=[RGBA_ORANGE], marker='o')
        ax.set_xticks([x_y_lim[0], cx, x_y_lim[1]])
        ax.set_yticks([x_y_lim[2], cy, x_y_lim[3]])
        scatters.append(sc)
    # wm = plt.get_current_fig_manager()
    # wm.window.state('zoomed')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.axis("off")
    plt.draw()
    plt.show(block=block)
    plt.pause(0.0001)
    return scatters


def save_pkl(file_path: str, data_dict: dict, ack: bool = True) -> None:
    file_obj = open(file_path, "wb")
    pickle.dump(data_dict, file_obj)
    file_obj.close()
    if ack:
        print('{} saved successfully'.format(file_path))
        # print(json_to_string(data_dict, indent=-1, sort_keys=False))
    return


def load_pkl(file_path: str, ack: bool = True) -> dict:
    file_obj = open(file_path, "rb")
    data_dict = pickle.load(file_obj)
    file_obj.close()
    if ack:
        print('{} loaded successfully'.format(file_path))
        # print(json_to_string(data_dict, indent=-1, sort_keys=False))
    return data_dict


def get_3d_orientation_from_2d_point(x: int, y: int, fx: int, R: np.array, float_pre: int = None):
    # R2 = np.copy(np.array(R, dtype=float))
    # intrinsic_mat = np.copy(np.array(intrinsic_mat, dtype=float))
    # the focal length in units of the x side length of a pixel
    # fx = intrinsic_mat[0][0]
    # calc orientation of camera: rotation @ [0,0,fx]  # 3x3 @ 3x1 -> 3,1.flatten()-> 3,
    # v = np.array([[x], [y], [fx]], dtype=float)  # shape 3,1
    v = np.array([[x], [y], [fx]], dtype=float)  # shape 3,1
    v = (R.T @ v).flatten()
    v /= np.linalg.norm(v)  # normed to len 1
    if float_pre is not None:
        v = np.round(v, float_pre)
    return v


def get_3d_position_from_R_and_T(R: np.array, T: np.array, float_pre: float) -> np.array:
    position = np.dot(-R.T, T.flatten())
    position = np.round(position, float_pre)
    return position


def move_figure(fig, x: int, y: int):
    """Move figure's upper left corner to pixel (x, y)"""
    try:
        x, y = int(x), int(y)
        new_geom = "+{}+{}".format(x, y)
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            manager = fig.canvas.manager
            manager.window.wm_geometry(new_geom)
        elif backend == 'WXAgg':
            fig.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            fig.canvas.manager.window.move(x, y)
    except (ValueError, Exception):
        print(WARNING.format('Failed Moving figure to ({},{})'.format(x, y), 'move_figure'))
        return
    return


def screen_dims():
    from PIL import ImageGrab
    try:
        img = ImageGrab.grab()
        window_w, window_h = img.size
    except (ValueError, Exception):
        window_w, window_h = 0, 0
        print(WARNING.format('Failed getting screen_dims', 'screen_dims'))
    return window_w, window_h


def move_plot(fig, where: str = 'top_left'):
    """
    :param fig: 
    :param where:
        top_right, top_center, top_left, bottom_right, bottom_center, bottom_left
    :return:
    """
    try:
        window_w, window_h = screen_dims()  # screen dims in pixels
        fig_w, fig_h = fig.get_size_inches() * fig.dpi  # fig dims in pixels
        task_bar_offset = 100  # about 100 pixels due to task bar

        x, y = 0, 0  # top_left: default

        if where == 'top_center':
            x, y = (window_w - fig_w) / 2, 0
        elif where == 'top_right':
            x, y = (window_w - fig_w), 0
        elif where == 'bottom_left':
            x, y = 0, window_h - fig_h - task_bar_offset
        elif where == 'bottom_center':
            x, y = (window_w - fig_w) / 2, window_h - fig_h - task_bar_offset
        elif where == 'bottom_right':
            x, y = (window_w - fig_w), window_h - fig_h - task_bar_offset
        # print('size', fig_h, fig_w)
        # print('x y', x, y)
        move_figure(fig=fig, x=x, y=y)
    except (ValueError, Exception):
        print(WARNING.format('Failed Moving figure to {}'.format(where), 'move_plot'))
        return
    return


def add_cube(axes, edge_len: int, add_corners_labels: bool):
    half_edge = int(edge_len / 2)
    xyz_bot_left = np.array([-half_edge, -half_edge, -half_edge], dtype=float)
    xyz_top_left = np.copy(xyz_bot_left)
    xyz_bot_right = np.copy(xyz_bot_left)
    xyz_bot_left_depth = np.copy(xyz_bot_left)
    xyz_top_left[1] += edge_len  # add just y
    xyz_bot_right[0] += edge_len  # add just x
    xyz_bot_left_depth[2] += edge_len  # add just z

    cube_4_edges = [xyz_bot_left, xyz_top_left, xyz_bot_right, xyz_bot_left_depth]
    plot_3d_cube(
        axes,
        cube_4_edges,
        label='cube(edge={})'.format(edge_len),
        add_corner_labels=add_corners_labels
    )
    return


def add_orientation_pyramids_to_cams_pyplot(axes, cameras_params, color, dist, height, width):
    pyramid_points_scatter_list, pyramid_points_lines_list = calculate_pyramid_points(cameras_params, dist,
                                                                                      height, width)
    for pyramid_points_scatter in pyramid_points_scatter_list:
        xs, ys, zs = pyramid_points_scatter[:, 0], pyramid_points_scatter[:, 1], pyramid_points_scatter[:, 2]
        axes.scatter(xs, ys, zs, c=color, s=10)
    for pyramid_points_lines in pyramid_points_lines_list:
        all_combinations = list(itertools.combinations(pyramid_points_lines, 2))
        for p1, p2 in all_combinations:
            axes.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color)
    return


def calculate_pyramid_points(cameras_params, dist, height, width) -> (list, list):
    """
    :param cameras_params:
    :param dist:
    :param height:
    :param width:
    :return:
    """
    offset_h = int(height / 2)
    offset_w = int(width / 2)
    pyramid_points_scatter_l, pyramid_points_lines_l = [], []
    for camera_params in cameras_params:
        p = camera_params.p
        R = camera_params.R
        fx = camera_params.get_focal_length_x()
        pyramid_offset_corners = [[-offset_w, -offset_h], [-offset_w, offset_h],
                                  [offset_w, offset_h], [offset_w, -offset_h]]
        directions = []
        for corner in pyramid_offset_corners:  # square
            # get orientation of camera: rotation @ [0,0,fx]  # 3x3 @ 3x1 -> 3,1.flatten()-> 3,
            # basically change the cx,cy of the image to get a square shape
            v = get_3d_orientation_from_2d_point(
                x=corner[0],
                y=corner[1],
                fx=fx,
                R=R,
                float_pre=None
            )
            directions.append(v)
        pyramid_corners_points = []
        for v in directions:
            pyramid_corner = p + (dist / np.linalg.norm(v)) * v
            pyramid_corners_points.append(pyramid_corner)
        corners_center = np.mean(pyramid_corners_points, axis=0)
        # add corners (no need for cam itself)
        pyramid_points_scatter = pyramid_corners_points + [corners_center]
        pyramid_points_scatter = np.array(pyramid_points_scatter, dtype=float)
        pyramid_points_scatter_l.append(pyramid_points_scatter)

        # connect all points with lines(add cam)
        pyramid_points_lines = [p] + pyramid_corners_points + [corners_center]
        pyramid_points_lines = np.array(pyramid_points_lines, dtype=float)
        pyramid_points_lines_l.append(pyramid_points_lines)
    return pyramid_points_scatter_l, pyramid_points_lines_l


def get_color_map(n: int) -> np.array:
    """
    :param n: how many colors
    :return: np array of size(n,4). each row in RGBA format
    """
    colors_map = cm.rainbow(np.linspace(0, 1, n))  # create color map
    return colors_map


def gstreamer_pipeline(
        cam_port,
        capture_width=640,  # 1280,
        capture_height=360,  # 720,
        display_width=640,  # 1280,
        display_height=360,  # 720,
        framerate=60,
        flip_method=0,
):
    # TODO use .format
    return (
            "nvarguscamerasrc sensor_id=" +
            str(cam_port) +
            " ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


class CameraParams:
    def __init__(self, mac: str, port: int, p: np.array, v: np.array, rotation_mat: np.array, intrinsic_mat: np.array,
                 distortion: np.array):
        """
        :param mac: mac of device the cam is on
        :param port: port of the camera
        :param p: origin point. shape=(3,)
        :param v: direction. shape=(3,)
        :param rotation_mat: shape=(3,3)
        :param intrinsic_mat: shape=(3,3)
        :param distortion: shape=(5,)  # TODO size is wrong. i think it's 1x5
        """
        self.mac = mac
        self.port = port
        self.p = p if isinstance(p, np.ndarray) else np.array(p, dtype=float)
        self.v = v if isinstance(v, np.ndarray) else np.array(v, dtype=float)
        self.R = rotation_mat if isinstance(rotation_mat, np.ndarray) else np.array(rotation_mat, dtype=float)
        self.intrinsic = intrinsic_mat if isinstance(intrinsic_mat, np.ndarray) else np.array(intrinsic_mat,
                                                                                              dtype=float)
        self.distortion = distortion  # TODO maybe cast to numpy array???
        return

    def __str__(self):
        to_str = 'CameraParams({}):'.format(self.get_id())
        to_str += '\n\t{}'.format(var_to_string(self.p, 'p', with_data=True))
        to_str += '\n\t{}'.format(var_to_string(self.v, 'v', with_data=True))
        to_str += '\n\t{}'.format(var_to_string(self.R, 'R', with_data=True))
        to_str += '\n\t{}'.format(var_to_string(self.intrinsic, 'intrinsic', with_data=True))
        to_str += '\n\t{}'.format(var_to_string(self.distortion, 'distortion', with_data=True))
        # to_str += '\n\t{}'.format(var_to_string(self.r, 'r', with_data=True))
        # to_str += '\n\t{}'.format(var_to_string(self.t, 't', with_data=True))
        return to_str

    def get_id(self, delimiter='/'):
        return '{}{}{}'.format(self.mac, delimiter, self.port)

    def get_focal_length_x(self):  # the focal length in units of the x side length of a pixel
        return self.intrinsic[0][0]

    def get_focal_length_y(self):  # the focal length in units of the y side length of a pixel
        return self.intrinsic[1][1]

    def get_center_x(self):
        return self.intrinsic[0][2]  # the x value of the image center

    def get_center_y(self):
        return self.intrinsic[1][2]  # the y value of the image center

    @staticmethod
    def read_calib_data(calib_dict: dict, rps_playing: list = None, debug: bool = False) -> list:
        cameras_params = []
        if debug:
            print('reading calib data into cameras params:')
        for mac, cam_ports_to_calib_data in calib_dict.items():
            if rps_playing is None or mac in rps_playing:
                for cam_port, calib_data in cam_ports_to_calib_data.items():
                    p = calib_data['position']
                    v = calib_data['orientation']
                    rotation_mat = calib_data['rotation_mat']
                    intrinsic_mat = calib_data['intrinsic_mat']
                    distortion = calib_data['distortion']
                    # r = calib_data['r']
                    # t = calib_data['t']
                    camera_params = CameraParams(mac, cam_port, p, v, rotation_mat, intrinsic_mat, distortion)
                    cameras_params.append(camera_params)
            else:
                if debug:
                    print('\tIgnoring {}'.format(mac))
        if debug:
            for camera_params in cameras_params:
                print(camera_params)
        return cameras_params

    @staticmethod
    def get_macs(cameras_params: list, ack: bool = False) -> list:
        macs_list = []
        for camera_params in cameras_params:
            if camera_params.mac not in macs_list:
                macs_list.append(camera_params.mac)
        if ack:
            print('macs_list={}'.format(macs_list))
        return macs_list

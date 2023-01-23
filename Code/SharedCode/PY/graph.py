import networkx as nx
from networkx.algorithms.distance_measures import center
import matplotlib.pyplot as plt
import itertools


# from SharedCode.PY.utilsScript import var_to_string


def get_cameras_list():
    cams = []
    for i in range(3):
        for j in range(2):
            cams.append('mac_{}_{}'.format(i, j))
    # cams = [
    #     'DC:A6:32:B8:1B:C6/0',
    #     'DC:A6:32:B8:1B:C6/1',
    #     'DC:A6:32:B8:1C:2F/0',
    #     'DC:A6:32:B8:1C:2F/1',
    #     'DC:A6:32:BE:8E:8D/0',
    #     'DC:A6:32:BE:8E:8D/1',
    #     'DC:A6:32:BE:B8:F9/0',
    #     'DC:A6:32:BE:B8:F9/1',
    # ]
    return cams


def build_nodes(cams):
    G = nx.Graph()
    for cam in cams:
        G.add_node(cam)
    return G


def show_graph(G, center_node=None, block: bool = True):
    pos = nx.spring_layout(G)
    if center_node is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=set(G.nodes - [center_node]))
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_nodes(G, pos, nodelist=[center_node], node_color='r')
    else:
        nx.draw(G, pos)

    nx.draw_networkx_labels(G, pos)
    plt.show(block=block)
    return


def save_graph(G, path, delimiter=':'):
    nx.write_edgelist(G, path=path, delimiter=delimiter)
    return


def load_graph(path, delimiter=':'):
    G = nx.read_edgelist(path=path, delimiter=delimiter)
    return G


def get_views_by_iterations():
    views = [
        ['mac_0_0', 'mac_0_1'],
        ['mac_0_0', 'mac_0_1', 'mac_1_0'],
        ['mac_1_0', 'mac_2_0'],
        ['mac_1_1', 'mac_2_0', 'mac_2_1']
    ]
    # views = [
    #     [
    #         'DC:A6:32:B8:1B:C6/1',
    #         'DC:A6:32:B8:1B:C6/0',
    #         'DC:A6:32:BE:B8:F9/1',
    #     ],
    #     [
    #         'DC:A6:32:BE:B8:F9/1',
    #         'DC:A6:32:BE:B8:F9/0',
    #         'DC:A6:32:B8:1C:2F/1',
    #         'DC:A6:32:B8:1C:2F/0',
    #         'DC:A6:32:B8:1B:C6/0',
    #         'DC:A6:32:B8:1B:C6/1',
    #     ],
    #     [
    #         'DC:A6:32:BE:B8:F9/0',
    #         'DC:A6:32:BE:B8:F9/1',
    #         'DC:A6:32:B8:1C:2F/1',
    #         'DC:A6:32:B8:1C:2F/0',
    #     ],
    #     [
    #         'DC:A6:32:B8:1B:C6/0',
    #         'DC:A6:32:B8:1B:C6/1',
    #         'DC:A6:32:BE:B8:F9/1',
    #         'DC:A6:32:BE:8E:8D/0',
    #         'DC:A6:32:BE:8E:8D/1',
    #     ],
    #     [
    #         'DC:A6:32:B8:1C:2F/1',
    #         'DC:A6:32:BE:B8:F9/1',
    #         'DC:A6:32:BE:B8:F9/0',
    #         'DC:A6:32:BE:8E:8D/0',
    #         'DC:A6:32:BE:8E:8D/1',
    #         'DC:A6:32:B8:1C:2F/0',
    #     ],
    #     [
    #         'DC:A6:32:B8:1C:2F/0',
    #         'DC:A6:32:B8:1C:2F/1',
    #         'DC:A6:32:BE:B8:F9/1',
    #         'DC:A6:32:BE:B8:F9/0',
    #         'DC:A6:32:BE:8E:8D/0',
    #         'DC:A6:32:BE:8E:8D/1',
    #     ],
    #     [
    #         'DC:A6:32:BE:8E:8D/0',
    #         'DC:A6:32:BE:8E:8D/1',
    #         'DC:A6:32:B8:1C:2F/0',
    #         'DC:A6:32:B8:1C:2F/1',
    #         'DC:A6:32:B8:1B:C6/0',
    #     ],
    #     [
    #         'DC:A6:32:BE:8E:8D/0',
    #         'DC:A6:32:BE:8E:8D/1',
    #         'DC:A6:32:B8:1C:2F/0',
    #         'DC:A6:32:B8:1C:2F/1',
    #         'DC:A6:32:B8:1B:C6/0',
    #     ],
    # ]
    return views


def create_edges(G, views):
    for view in views:
        n_choose_2 = list(itertools.combinations(view, 2))
        for cam1, cam2 in n_choose_2:
            e = (cam1, cam2)
            if not G.has_edge(*e):
                G.add_edge(*e)
    # for view in views:
    #     cam1 = view[0]
    #     for i in range(1, len(view)):
    #         cam2 = view[i]
    #         e = (cam1, cam2)
    #         if not G.has_edge(*e):
    #             G.add_edge(*e)
    return


def get_center(G):
    """
    if more than one center found, get the highest degree one
    :param G:
    :return:
    """
    centers = center(G)  # could be more than 1
    best_center = None
    if len(centers) > 1:
        max_degree = -1
        for c in centers:
            c_degree = len(list(G.neighbors(c)))
            print('{} has {} neighbours'.format(c, c_degree))
            if c_degree > max_degree:
                print('new best {}'.format(c))
                max_degree = c_degree
                best_center = c
    else:
        best_center = centers[0]
    return best_center


def main():
    cams = get_cameras_list()
    g = build_nodes(cams)
    # show_graph(G=g)
    joint_views = get_views_by_iterations()
    create_edges(G=g, views=joint_views)
    ground_cam = get_center(g)
    print('best gr cam {}'.format(ground_cam))
    # show_graph(g, ground_cam)

    # bfs_edges = list(nx.bfs_edges(g, source=ground_cam))
    # print('bfs_edges: {}'.format(bfs_edges))
    t = nx.bfs_tree(g, source=ground_cam)
    t_edges = list(nx.bfs_edges(t, source=ground_cam))
    print('t edges:   {}'.format(t_edges))
    # show_graph(t, None)
    # show_graph(t, center(g)[0])

    print('extrinsic pairs:')
    print('\tcalibrating (relative grounds):')
    for i, (gr, other) in enumerate(t_edges):
        print('pair {}: gr {}, other {}'.format(i, gr, other))

    print('\tget to real ground by bfs short path to other:')
    all_paths_edges = nx.shortest_path(G=g, source=ground_cam)
    for i, (other, path_list) in enumerate(all_paths_edges.items()):
        print('{})other {} - path list {}'.format(i, other, path_list))
        if len(path_list) == 1:  # ground cam
            pass
        elif len(path_list) == 2:  # ground neighbour - just save
            pass
        elif len(path_list) > 2:  # need to use bfs path to set other relative to ground cam
            pass

    show_graph(G=g, center_node=ground_cam, block=True)
    # t = nx.bfs_tree(g, center(g)[0])
    # show_graph(G=t)
    # path, d = 'nx_graph', ':'
    # save_graph(G=g, path=path, delimiter=d)
    # g2 = load_graph(path=path, delimiter=d)
    # show_graph(G=g2)
    return


if __name__ == '__main__':
    main()

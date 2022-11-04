"""Useful functions for geometrical computations in the maze space"""

import numpy as np
from shapely.geometry.polygon import Polygon
from data_generation.utils.graph import build_graph

#
l = [(1 - 1 / (np.sqrt(2) + 1)) / 2,
     (1 + 1 / (np.sqrt(2) + 1)) / 2,
     1,
     3 / 2 - 1 / (2 * (np.sqrt(2) + 1))]  # used to define octogons

def create_octogon_from_point(p): ##Todo move to utils or geom
    """The input point is the bottom left corner of a 1x1 square.
    Returns octagon inscribed in the square"""

    pol = Polygon([np.add(p, [l[0], 0]), np.add(p, [l[1], 0]), np.add(p, [1, l[0]]),
                   np.add(p, [1, l[1]]), np.add(p, [l[1], 1]), np.add(p, [l[0], 1]),
                   np.add(p, [0, l[1]]), np.add(p, [0, l[0]])])

    return pol

def create_connecting_square(c, c_1):

    """creates square objects that connect the octagonal tiles"""

    square = None
    if np.linalg.norm(np.asarray(c) - np.asarray(c_1)) > 1:
        if c[0] > c_1[0] and c[1] > c_1[1]:
            square = Polygon([np.add(c, [0, l[0]]), np.add(c, [l[0], 0]),
                              np.add(c, [0, -l[0]]), np.add(c, [-l[0], 0])])
        elif c[0] > c_1[0] and c[1] < c_1[1]:
            square = Polygon([np.add(c, [0, l[1]]), np.add(c, [- l[0], 1]),
                              np.add(c, [0, l[3]]),
                              np.add(c, [l[0], 1])])
        elif c[0] < c_1[0] and c[1] > c_1[1]:
            square = Polygon([np.add(c, [l[1], 0]), np.add(c, [1, l[0]]),
                              np.add(c, [l[3], 0]),
                              np.add(c, [1, - l[0]])])
        elif c[0] < c_1[0] and c[1] < c_1[1]:
            square = Polygon([np.add(c, [l[1], 1]), np.add(c, [1, l[3]]),
                              np.add(c, [l[3], 1]),
                              np.add(c, [1, l[1]])])
        else:
            print("WARNING: empty square object!")

    return square


def euclidean_distance(x, c):
    """compute Euclidean distances between array of positions x and array of positions c (typically rat's position and place fields centers)"""

    c_resh = np.repeat(c[:, np.newaxis, :], x.shape[1], axis=1) #reshape
    mat = x[:, :, np.newaxis] - c_resh
    d = np.linalg.norm(mat, axis=0, ord = 2)

    return d

def graph_distance(maze, maze_config, x, c):
    """compute graph distances between array of positions x and array of positions c (typically rat's position and place fields centers)"""


    # graph distances
    cellList = np.asarray(maze.cellList[maze_config]).T + 0.5

    # map positions to maze cells
    x_tiles_mapping = shortest_distance_idx(x, cellList)
    c_tiles_mapping = shortest_distance_idx(c, cellList)


    #find shortest path between cells
    graph = build_graph(maze.edgeList[maze_config])
    tileArray = np.array(maze.cellList[maze_config])
    xTiles = tileArray[x_tiles_mapping]
    cTiles = tileArray[c_tiles_mapping]

    cTiles_resh = np.repeat(cTiles[np.newaxis, :,  :], xTiles.shape[0], axis=0)  # reshape
    d = np.ones([len(xTiles), len(cTiles)]) * np.inf
    idx_candidates = np.where(np.linalg.norm(xTiles[:, np.newaxis, :]-cTiles_resh, axis=-1) <= np.sqrt(2))
    idx_x_compute = []
    idx_c_compute = []
    for i in range(len(idx_candidates[0])):
        idx_x = idx_candidates[0][i]
        idx_c = idx_candidates[1][i]

        if (xTiles[idx_x] in np.array(graph[str(list(cTiles[idx_c]))])) or xTiles[idx_x].all() == cTiles[idx_c].all(): #check if x and c in adjacent or same tiles
            idx_x_compute.append(idx_x)
            idx_c_compute.append(idx_c)

    d[idx_x_compute, idx_c_compute] = np.linalg.norm(x[:, idx_x_compute]-c[:, idx_c_compute], axis=0, ord = 2)

    return d

def shortest_distance_idx(pos_array, ref_array):
    """computes distances between points contained in two input arrays. For each point in pos_array,
    returns the indice of the closets point in ref_array."""

    ref_array_resh = np.repeat(ref_array[:, :, np.newaxis], pos_array.shape[1], axis=2)
    mat = pos_array[:, np.newaxis, :] - ref_array_resh
    d_to_ref = np.linalg.norm(mat, axis=0)
    closest_idx = np.argmin(d_to_ref, axis=0)

    return closest_idx
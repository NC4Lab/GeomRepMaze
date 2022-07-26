import numpy as np
from shapely.geometry.polygon import Polygon


l = [(1 - 1 / (np.sqrt(2) + 1)) / 2, (1 + 1 / (np.sqrt(2) + 1)) / 2, 1,
     3 / 2 - 1 / (2 * (np.sqrt(2) + 1))]  # used to define octogons

def create_octogon_from_point(p): ##Todo move to utils or geom
    """The input point is the bottom left corner of a 1x1 square.
    Return octogon contained in that square"""

    l = [(1 - 1 / (np.sqrt(2) + 1)) / 2, (1 + 1 / (np.sqrt(2) + 1)) / 2, 1,
              3 / 2 - 1 / (2 * (np.sqrt(2) + 1))]  # used to define octogons

    pol = Polygon([np.add(p, [l[0], 0]), np.add(p, [l[1], 0]), np.add(p, [1, l[0]]),
                   np.add(p, [1, l[1]]), np.add(p, [l[1], 1]), np.add(p, [l[0], 1]),
                   np.add(p, [0, l[1]]), np.add(p, [0, l[0]])])

    return pol


def create_connecting_square(c, c_1):
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
            print("WARNING empty square object!!")

    return square


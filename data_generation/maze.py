import numpy as np
import math
from matplotlib.path import Path
from matplotlib import pyplot as plt
from shapely.ops import unary_union
from shapely.geometry.polygon import Polygon


def create_octogon_from_point(p): ##Todo move to utils or geom
    "The input point is the bottom left corner of a 1x1 square. Return an octogon contained in that square "

    l = [(1 - 1 / (np.sqrt(2) + 1)) / 2, (1 + 1 / (np.sqrt(2) + 1)) / 2, 1,
              3 / 2 - 1 / (2 * (np.sqrt(2) + 1))]  # used to define octogons

    pol = Polygon([np.add(p, [l[0], 0]), np.add(p, [l[1], 0]), np.add(p, [1, l[0]]),
                   np.add(p, [1, l[1]]), np.add(p, [l[1], 1]), np.add(p, [l[0], 1]),
                   np.add(p, [0, l[1]]), np.add(p, [0, l[0]])])

    return pol


"Class to define a Square Maze with obstacles"
class Maze():

    def __init__(self, mazeSettings):
        #initialize 2D space
        self.N = mazeSettings["size"]  # Maze width
        self.octoMazeBool = mazeSettings["octogonalMaze"] #a boolean to choose betwen octgonal or square maze grid
        self.home = mazeSettings["home"]
        self.goals = None

        self.mazeRes = mazeSettings["resolution"]  # pixels per tile
        #Maze config initialization
        self.squareMaze = np.zeros((self.N, self.N), dtype=bool)
        self.octoMaze = None
        self.mazeFlags = np.zeros([self.mazeRes*self.N, self.mazeRes*self.N], dtype = bool)


    def create_maze(self, mazeCells):
        """create a maze"""
        if self.octoMazeBool:
            self.createOctoMaze(mazeCells)
        else:
            self.createSquareMaze(mazeCells)

    def createSquareMaze(self, mazeCells, home = None, goals = None):
        self.home = home
        self.goals = goals

        for i in range(len(mazeCells)):
            self.squareMaze[tuple(np.asarray(mazeCells[i]).T)] = True #set routes

        mazeFlags = np.repeat(self.squareMaze, self.mazeRes, axis = 1)
        self.mazeFlags = np.repeat(mazeFlags, self.mazeRes, axis = 0).T

    def createOctoMaze(self, mazeCells):
        nb_path = len(mazeCells)
        polygonList = []
        l = [(1 - 1 / (np.sqrt(2) + 1)) / 2, (1 + 1 / (np.sqrt(2) + 1)) / 2, 1,
             3 / 2 - 1 / (2 * (np.sqrt(2) + 1))]  # used to define octogons
        plt.figure()

        for i in range(nb_path):

            # create octogons
            for j in range(len(mazeCells[i])):
                c = mazeCells[i][j][:]
                pol = create_octogon_from_point(c)
                polygonList.append(pol)

                #create squares
                square = []
                if j+1 < len(mazeCells[i]):
                    c_1 = mazeCells[i][j + 1][:]
                    #TODO add function for square
                    if np.linalg.norm(np.asarray(c) - np.asarray(c_1))>1:

                        if c[0] > c_1[0] and c[1] > c_1[1]:
                            square = Polygon([np.add(c, [0, l[0]]), np.add(c, [l[0], 0]),
                                              np.add(c, [0, -l[0]]), np.add(c, [-l[0],0])])
                        elif c[0] > c_1[0] and c[1] < c_1[1]:
                            square = Polygon([np.add(c, [0, l[1]]), np.add(c, [- l[0], 1]),
                                              np.add(c, [0, l[3]]),
                                              np.add(c, [l[0], 1])])
                        elif c[0] < c_1[0] and c[1] > c_1[1]:
                            square = Polygon([np.add(c, [l[1], 0]), np.add(c, [1,  l[0]]),
                                              np.add(c, [l[3],  0]),
                                              np.add(c, [1, - l[0]])])
                        elif c[0] < c_1[0] and c[1] < c_1[1]:
                            square = Polygon([np.add(c, [l[1], 1]), np.add(c, [1, l[3]]),
                                              np.add(c, [l[3], 1]),
                                              np.add(c, [1, l[1]])])
                        else:
                            print("WARNING empty square object!!")

                        polygonList.append(square)

        #merge polygons and convert to path object
        self.octoMaze = Path(np.asarray(unary_union(polygonList).exterior.xy).T)

        #create binary flags for each pixels (either in or out maze tiles)
        M = self.N*self.mazeRes
        xv, yv = np.meshgrid(np.linspace(0, self.N, M), np.linspace(0, self.N, M))
        binMaze = self.octoMaze.contains_points(np.hstack((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis])))
        self.mazeFlags = binMaze.reshape(M, M)

    def isInMaze(self, x, y):
        """checks if the point in input is in the maze"""

        #Octogonal maze
        if self.octoMazeBool:
            return self.octoMaze.contains_point([x, y])

        #Square maze
        x_cor = int(np.floor(x))
        y_cor = int(np.floor(y))

        if x_cor == 7:
            x_cor = 6
        if y_cor == 7:
            y_cor = 6

        if y_cor < 0 or x_cor < 0 or y_cor > 7 or x_cor > 7:
            return False
        if self.squareMaze[x_cor, y_cor] == 1:
            return True
        else:
            return False

    def get_adjacent_points(self, x_coor, y_coor, d):
        """returns points adjacent to a point that are in the maze, given a displacement d"""

        list_adjcoorX = []
        list_adjcoorY = []

        for x, y in [(x_coor + i, y_coor + j) for i in (-d, 0, d) for j in (-d, 0, d) if i != 0 or j != 0]:
            if self.isInMaze(x, y):
                list_adjcoorX.append(x)
                list_adjcoorY.append(y)

        return np.asarray(list_adjcoorX), np.asarray(list_adjcoorY)


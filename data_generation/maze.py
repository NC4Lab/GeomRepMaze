import numpy as np
import math
from matplotlib.path import Path
from matplotlib import pyplot as plt
from shapely.ops import unary_union
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point




"Class to define a Square Maze with obstacles"
class Maze():

    def __init__(self, N, octoMazeBool = False, MazeRes = 100):
        #initialize 2D space
        self.octoMazeBool = octoMazeBool #a boolean to choose betwen octgonal or square maze grid
        self.N = N #Maze width

        self.squareMaze = np.zeros((self.N, self.N), dtype=bool) #contain routes (1 = route, 0 = non-route)
        self.octoMaze = None
        self.home = None
        self.goals = None
        self.MazeRes = MazeRes #pixels per tile
        self.mazeFlags = np.zeros([self.MazeRes*N, self.MazeRes*N], dtype = bool)

    def create_maze(self, mazeCells, home = None, goals = None):
        if self.octoMazeBool:
            self.createOctoMaze(mazeCells)
        else:
            self.createSquareMaze(mazeCells)

    def createSquareMaze(self, mazeCells, home = None, goals = None):
        #create maze routes

        self.home = home
        self.goals = goals

        for i in range(len(mazeCells)):
            self.squareMaze[tuple(np.asarray(mazeCells[i]).T)] = True #set routes

        mazeFlags = np.repeat(self.squareMaze, self.MazeRes, axis = 1)
        self.mazeFlags = np.repeat(mazeFlags, self.MazeRes, axis = 0).T #[::-1]

    def createOctoMaze(self, mazeCells):
        nb_path = len(mazeCells)
        polygonList = []
        plt.figure()
        for i in range(nb_path):
            for j in range(len(mazeCells[i])):
                c = mazeCells[i][j][:]
                pol = Polygon([np.add(c, [0.25, 0]), np.add(c, [0.75, 0]), np.add(c, [1, 0.25]),
                             np.add(c, [1,0.75]), np.add(c, [0.75, 1]), np.add(c, [0.25, 1]),
                             np.add(c,[0, 0.75]), np.add(c,[0, 0.25])])
                polygonList.append(pol)


                square = []
                if j+1 < len(mazeCells[i]):
                    c_1 = mazeCells[i][j + 1][:]

                    if np.linalg.norm(np.asarray(c) - np.asarray(c_1))>1:

                        if c[0] > c_1[0] and c[1] > c_1[1]:
                            square = Polygon([np.add(c, [0, 0.25]), np.add(c, [0.25, 0]),
                                              np.add(c, [0, -0.25]), np.add(c, [-0.25,0])])
                        elif c[0] > c_1[0] and c[1] < c_1[1]:
                            square = Polygon([np.add(c, [0, 0.75]), np.add(c, [-0.25, 1]),
                                              np.add(c, [0, 1.25]),
                                              np.add(c, [0.25, 1])])
                        elif c[0] < c_1[0] and c[1] > c_1[1]:
                            square = Polygon([np.add(c, [0.75, 0]), np.add(c, [1, 0.25]),
                                              np.add(c, [1.25, 0]),
                                              np.add(c, [1, -0.25])])
                        elif c[0] < c_1[0] and c[1] < c_1[1]:
                            square = Polygon([np.add(c, [0.75, 1]), np.add(c, [1, 1.25]),
                                              np.add(c, [1.25, 1]),
                                              np.add(c, [1, 0.75])])
                        else: print("WARNING empty square object")

                        polygonList.append(square)

                #plt.plot(*u.exterior.xy)

        self.octoMaze = Path(np.asarray(unary_union(polygonList).exterior.xy).T)
        M = self.N*self.MazeRes
        xv, yv = np.meshgrid(np.linspace(0, self.N, M), np.linspace(0, self.N, M))

        binMaze = a = self.octoMaze.contains_points(np.hstack((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis])))
        self.mazeFlags = binMaze.reshape(M, M)

        """u = unary_union(polygonList)
        plt.plot(*u.exterior.xy)
        plt.show()"""


    def isInMaze(self, x, y):
        #check if point in the maze
        if self.octoMazeBool:
            return self.octoMaze.contains_point([x, y])

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

    def get_adjacent_cells(self, x_coor, y_coor, d):
        #returns cells adjacent to a point that are in the maze

        list_adjcoorX = []
        list_adjcoorY = []

        for x, y in [(x_coor + i, y_coor + j) for i in (-d, 0, d) for j in (-d, 0, d) if i != 0 or j != 0]:
            if self.isInMaze(x, y):
                list_adjcoorX.append(x)
                list_adjcoorY.append(y)

        return np.asarray(list_adjcoorX), np.asarray(list_adjcoorY)


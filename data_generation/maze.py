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
        #self.homes = mazeSettings["homes"]
        #self.goals = None
        self.trial = mazeSettings["trial_nb"]
        self.mazeRes = mazeSettings["resolution"]  # pixels per tile

        #Maze config initialization
        self.fullSquareMaze = np.zeros((self.N, self.N), dtype=bool)
        self.trialSquareMaze = np.zeros((self.N, self.N), dtype=bool)

        self.fullOctoMaze = None
        self.trialOctoMaze = None
        self.fullMazeFlags = np.zeros([self.mazeRes*self.N, self.mazeRes*self.N], dtype = bool)
        self.trialMazeFlags = np.zeros([self.mazeRes*self.N, self.mazeRes*self.N], dtype = bool)


        self.createFullMaze(mazeSettings)

    def createFullMaze(self, mazeSettings):
        """create a maze"""
        if self.octoMazeBool:
            self.createFullOctoMaze(mazeSettings)
        else:
            self.createFullSquareMaze(mazeSettings)

    def createFullSquareMaze(self, mazeSettings):
        mazeCells = mazeSettings["mazeCells"]

        #for i in range(len(trialCells)):
        self.fullSquareMaze[tuple(np.asarray(mazeCells).T)] = True

        mazeFlags = np.repeat(self.fullSquareMaze, self.mazeRes, axis=1)
        self.fullMazeFlags = np.repeat(mazeFlags, self.mazeRes, axis=0).T

    def createFullOctoMaze(self, mazeSettings):
        mazeCells = mazeSettings["cellList"]
        mazeEdges = mazeSettings["edgeList"]

        #nb_path = len(mazeCells)
        polygonList = []
        l = [(1 - 1 / (np.sqrt(2) + 1)) / 2, (1 + 1 / (np.sqrt(2) + 1)) / 2, 1,
             3 / 2 - 1 / (2 * (np.sqrt(2) + 1))]  # used to define octogons

        # create octogons
        for i in range(len(mazeCells)):
            for j in range(len(mazeCells[i])):
                c = mazeCells[i][j]
                pol = create_octogon_from_point(c)
                polygonList.append(pol)

        #create squares
        square = []
        for i in range(len(mazeEdges)):
            for j in range(len(mazeEdges[i])):
                c = mazeEdges[i][j][0]
                c_1 = mazeEdges[i][j][1]
                # TODO add function for square

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

                    polygonList.append(square)

        #merge polygons and convert to path object
        self.fullOctoMaze = Path(np.asarray(unary_union(polygonList).exterior.xy).T)

        M = self.N * self.mazeRes
        xv, yv = np.meshgrid(np.linspace(0, self.N, M), np.linspace(0, self.N, M))

        binMaze = self.fullOctoMaze.contains_points(np.hstack((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis])))
        self.fullMazeFlags = binMaze.reshape(M, M)

    def createTrialMaze(self, mazeSettings, trial = None):

        if self.octoMazeBool:
            self.createTrialOctoMaze(mazeSettings, trial)
        else:
            self.createTrialSquareMaze(mazeSettings, trial)

    def createTrialSquareMaze(self, mazeSettings, trial):
        if trial is None:
            trial = mazeSettings["trial_nb"]

        trialCells = mazeSettings["edgeList"][trial]

        self.trialSquareMaze = np.fill([self.N, len(trialCells)], False)
        self.trialSquareMaze[tuple(np.asarray(trialCells).T)] = True

        mazeFlags = np.repeat(self.trialSquareMaze, self.mazeRes, axis=1)
        self.trialMazeFlags = np.repeat(mazeFlags, self.mazeRes, axis=0).T

    def createTrialOctoMaze(self, mazeSettings, trial):
        if trial is None:
            trial = mazeSettings["trial_nb"]

        trialCells = mazeSettings["cellList"][mazeSettings["trial_nb"]]
        trialEdges = mazeSettings["edgeList"][mazeSettings["trial_nb"]]

        polygonList = []
        l = [(1 - 1 / (np.sqrt(2) + 1)) / 2, (1 + 1 / (np.sqrt(2) + 1)) / 2, 1,
             3 / 2 - 1 / (2 * (np.sqrt(2) + 1))]  # used to define octogons

        # create octogons
        for i in range(len(trialCells)):
            c = trialCells[i]
            pol = create_octogon_from_point(c)
            polygonList.append(pol)

        # create squares
        square = []
        for i in range(len(trialEdges)):
            c = trialEdges[i][0]
            c_1 = trialEdges[i][1]
            # TODO add function for square

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

                polygonList.append(square)

        # merge polygons and convert to path object
        self.trialOctoMaze = Path(np.asarray(unary_union(polygonList).exterior.xy).T)

            # create binary flags for each pixels (either in or out maze tiles)
        M = self.N * self.mazeRes
        xv, yv = np.meshgrid(np.linspace(0, self.N, M), np.linspace(0, self.N, M))
        binMaze = self.trialOctoMaze.contains_points(np.hstack((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis])))
        self.trialMazeFlags = binMaze.reshape(M, M)

    def isInMaze(self, x, y, mode = "trial"):
        """checks if the point in input is in the maze"""

        #Octogonal maze
        if self.octoMazeBool:
            if mode == "trial":
                return self.trialOctoMaze.contains_point([x, y])
            elif mode == "full":
                return self.fullOctoMaze.contains_point([x, y])
            else:
                print("Error: specify a valid mode")


        #Square maze
        x_cor = int(np.floor(x))
        y_cor = int(np.floor(y))

        if x_cor == 7:
            x_cor = 6
        if y_cor == 7:
            y_cor = 6

        if y_cor < 0 or x_cor < 0 or y_cor > 7 or x_cor > 7:
            return False

        if mode == "trial":
            return self.trialSquareMaze[x_cor, y_cor] == 1

        elif mode == "full":
            return self.fullSquareMaze[x_cor, y_cor] == 1
        else:
            print("Error: specify a valid mode")

    def get_adjacent_points(self, x_coor, y_coor, d):
        """returns points adjacent to a point that are in the maze, given a displacement d"""

        list_adjcoorX = []
        list_adjcoorY = []

        for x, y in [(x_coor + i, y_coor + j) for i in (-d, 0, d) for j in (-d, 0, d) if i != 0 or j != 0]:
            if self.isInMaze(x, y, mode = "trial"):
                list_adjcoorX.append(x)
                list_adjcoorY.append(y)

        return np.asarray(list_adjcoorX), np.asarray(list_adjcoorY)

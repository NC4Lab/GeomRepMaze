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
        self.nb_of_trials = mazeSettings["nb_of_trials"]
        self.mazeRes = mazeSettings["resolution"]  # pixels per tile
        self.cellList = mazeSettings["cellList"]
        self.edgeList = mazeSettings["edgeList"]
        self.nodeList = mazeSettings["nodeList"]
        #Maze config initialization
        self.fullSquareMaze = np.zeros((self.N, self.N), dtype=bool)
        self.trialSquareMaze = np.zeros((self.nb_of_trials,self.N, self.N), dtype=bool)

        self.fullOctoMaze = None
        self.fullOctoMaze = None
        self.trialOctoMaze = []
        self.fullMazeFlags = np.zeros([self.nb_of_trials, self.mazeRes*self.N, self.mazeRes*self.N], dtype = bool)
        self.trialMazeFlags = np.zeros([self.nb_of_trials, self.mazeRes*self.N, self.mazeRes*self.N], dtype = bool)


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

    def createTrialSquareMaze(self, mazeSettings):

        for i in range(mazeSettings["nb_of_trials"]):
            trialCells = mazeSettings["edgeList"][i]
            #self.trialSquareMaze[i, :, :] = np.fill([self.N, len(trialCells)], False)

            self.trialSquareMaze[i, tuple(np.asarray(trialCells).T)] = True

            mazeFlags = np.repeat(self.trialSquareMaze, self.mazeRes, axis=1)
            self.trialMazeFlags[i, :, :] = np.repeat(mazeFlags, self.mazeRes, axis=0).T

    def createTrialOctoMaze(self, mazeSettings, trial):

        for n in range(mazeSettings["nb_of_trials"]):

            trialCells = mazeSettings["cellList"][n]
            trialEdges = mazeSettings["edgeList"][n]

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
            self.trialOctoMaze.append(Path(np.asarray(unary_union(polygonList).exterior.xy).T))

                # create binary flags for each pixels (either in or out maze tiles)
            M = self.N * self.mazeRes
            xv, yv = np.meshgrid(np.linspace(0, self.N, M), np.linspace(0, self.N, M))
            binMaze = self.trialOctoMaze[n].contains_points(np.hstack((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis])))
            self.trialMazeFlags[n, :, :] = binMaze.reshape(M, M)

    def isInMaze(self, x, y, trial = None, mode = "trial"):
        """checks if the point in input is in the maze"""

        #Octogonal maze
        if self.octoMazeBool:
            if mode == "trial" and np.size(x) > 1:
                return self.trialOctoMaze[trial].contains_points(np.array([x, y]).T)
            elif mode == "trial" and np.size(x) == 1:
                return self.trialOctoMaze[trial].contains_point([x, y])
            elif mode == "full" and np.size(x) > 1:
                return self.fullOctoMaze.contains_points(np.array([x, y]).T)
            elif mode == "full" and np.size(x) == 1:
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
            return self.trialSquareMaze[trial, x_cor, y_cor] == 1

        elif mode == "full":
            return self.fullSquareMaze[x_cor, y_cor] == 1
        else:
            print("Error: specify a valid mode")

    def get_adjacent_points(self, x_coor, y_coor, d, trial):
        """returns points adjacent to a point that are in the maze, given a displacement d"""
        K = 360 #number of adjacet points

        dx = d*np.cos(2*np.pi/K*(np.arange(K)+1))
        dy = d*np.sin(2*np.pi/K*(np.arange(K)+1))
        x_prov = x_coor + dx
        y_prov = y_coor + dy

        inMazeIdx = self.isInMaze(x_prov, y_prov, mode="trial", trial=trial)

        x = x_prov[inMazeIdx]
        y = y_prov[inMazeIdx]
        """for x, y in [(x_coor + i, y_coor + j) for i in (-d, 0, d) for j in (-d, 0, d) if i != 0 or j != 0]:
            if self.isInMaze(x, y, mode = "trial"):
                list_adjcoorX.append(x)
                list_adjcoorY.append(y)"""

        return x, y


from matplotlib.path import Path
from shapely.ops import unary_union
from data_generation.utils.geom_utils import *
from data_generation.utils.graph import BFS_SP



"Class to define a Maze platform with different configurations"
class Maze():

    def __init__(self, mazeSettings):
        self.N = mazeSettings["size"]  # Maze width
        self.octoMazeBool = mazeSettings["octogonalMaze"] #a boolean to choose betwen octgonal or square maze grid

        self.nb_of_trials= mazeSettings["nb_of_trials"]
        self.mazeRes = mazeSettings["resolution"]  # pixels width per cell

        self.nodeList = mazeSettings["nodeList"]
        self.edgeList = self.get_edge_list()
        self.cellList = self.get_cell_list()

        self.fullSquareMaze = np.zeros((self.N, self.N), dtype=bool)
        self.trialSquareMaze = np.zeros((self.nb_of_trials,self.N, self.N), dtype=bool)

        self.fullOctoMaze = None
        self.trialOctoMaze = []
        self.fullMazeFlags = np.zeros([self.nb_of_trials, self.mazeRes*self.N, self.mazeRes*self.N], dtype = bool)
        self.trialMazeFlags = np.zeros([self.nb_of_trials, self.mazeRes*self.N, self.mazeRes*self.N], dtype = bool)


        self.createFullMaze(mazeSettings)
        self.connectedNodes = self.get_connected_nodes()

    def get_edge_list(self):
        keyList = ["A", "C", "D", "E"]
        edge_list = []

        for i, n in enumerate(self.nodeList):
            sublist = []
            t0 = np.array(n["B"])

            for k in keyList:
                t1 = np.array(n[str(k)])
                while list(t0) != list(t1):
                    t1_n = t1 + np.sign(t0 - t1)
                    sublist.append([list(t1), list(t1_n)])
                    t1 = t1_n

            edge_list.append(sublist)
        return edge_list

    def get_cell_list(self):
        cell_list = []
        for i, c in enumerate(self.edgeList):

            c = np.array(c).reshape((int(np.array(c).size/2), 2))
            cell_list.append(np.unique(c, axis=0))
        return cell_list

    def get_connected_nodes(self):
        connectedNodes = []
        edgeTiles = []

        graph = build_graph(self.edgeList[0])
        keys = sorted(self.nodeList[0].keys())
        nodeList = self.nodeList[0]

        for j in range(len(nodeList) - 1):
            for k in np.arange(j + 1, len(nodeList)):
                path = BFS_SP(graph, nodeList[keys[j]], nodeList[keys[k]])
                common_el = 0
                for n in nodeList.values():
                    if (np.array(path) == n).all(axis=-1).any():
                        common_el = common_el + 1
                if common_el == 2:
                    connectedNodes.append([keys[j], keys[k]])
                    edgeTiles.append(path)

        return connectedNodes


    def createFullMaze(self, mazeSettings):
        """create full maze, merging all maze configurations"""
        if self.octoMazeBool:
            self.createFullOctoMaze(mazeSettings)
        else:
            self.createFullSquareMaze(mazeSettings)

    def createFullSquareMaze(self, mazeSettings):
        mazeCells = mazeSettings["mazeCells"]
        self.fullSquareMaze[tuple(np.asarray(mazeCells).T)] = True

        mazeFlags = np.repeat(self.fullSquareMaze, self.mazeRes, axis=1)
        self.fullMazeFlags = np.repeat(mazeFlags, self.mazeRes, axis=0).T

    def createFullOctoMaze(self, mazeSettings):
        mazeCells = self.cellList
        mazeEdges = self.edgeList

        polygonList = []

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
                square = create_connecting_square(c, c_1)
                if square is not None:
                    polygonList.append(square)

        #merge polygons and convert to path object
        self.fullOctoMaze = Path(np.asarray(unary_union(polygonList).exterior.xy).T)

        self.fullMazeFlags = self.polygon_to_flags(self.fullOctoMaze, self.mazeRes)

    def createTrialMaze(self, mazeSettings, trial = None):

        if self.octoMazeBool:
            self.createTrialOctoMaze(mazeSettings, trial)
        else:
            self.createTrialSquareMaze(mazeSettings, trial)

    def createTrialSquareMaze(self, mazeSettings):

        for i in range(mazeSettings["nb_of_trials"]):
            trialCells = mazeSettings["edgeList"][i]
            self.trialSquareMaze[i, tuple(np.asarray(trialCells).T)] = True
            mazeFlags = np.repeat(self.trialSquareMaze, self.mazeRes, axis=1)
            self.trialMazeFlags[i, :, :] = np.repeat(mazeFlags, self.mazeRes, axis=0).T

    def createTrialOctoMaze(self, mazeSettings, trial):

        for n in range(mazeSettings["nb_of_trials"]):

            trialCells = self.cellList[n]
            trialEdges = self.edgeList[n]

            polygonList = []

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

                square = create_connecting_square(c, c_1)
                if square is not None:
                    polygonList.append(square)

            # merge polygons and convert to path object
            self.trialOctoMaze.append(Path(np.asarray(unary_union(polygonList).exterior.xy).T))

            # create binary flags for each pixels (either in or out maze tiles)
            self.trialMazeFlags[n, :, :] = self.polygon_to_flags(self.trialOctoMaze[n], self.mazeRes)

    def isInMaze(self, x, y, trial = None, mode = "trial"):
        """checks if the point (x, y) in input is in the maze"""

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

    def get_adjacent_points(self, x_coor, y_coor, d, angular_res, trial):
        """returns points adjacent to a point that are in the maze, given a displacement d"""
        K = angular_res #number of adjacet points

        dx = d*np.cos(2*np.pi/K*(np.arange(K)+1))
        dy = d*np.sin(2*np.pi/K*(np.arange(K)+1))
        x_prov = x_coor + dx
        y_prov = y_coor + dy

        inMazeIdx = self.isInMaze(x_prov, y_prov, mode="trial", trial=trial)

        x = x_prov[inMazeIdx]
        y = y_prov[inMazeIdx]

        return x, y

    def get_cell_flags(self, cell_center):
        cell_pol = create_octogon_from_point(cell_center)
        cell_pol = Path(np.asarray(cell_pol.exterior.xy).T)

        # create binary flags for each pixels (either in or out maze tiles)
        flags = self.polygon_to_flags(cell_pol, self.mazeRes)

        return flags

    def polygon_to_flags(self, polygon, res):
        M = self.N * res
        xv, yv = np.meshgrid(np.linspace(0, self.N, M), np.linspace(0, self.N, M))
        flags = polygon.contains_points(
            np.hstack((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis]))).reshape((M, M))

        flags = flags.reshape(M,M)

        return flags
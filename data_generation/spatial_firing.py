import numpy as np
import random
import math
from matplotlib import pyplot as plt
from data_generation.graph import build_graph, BFS_SP, compute_path_distance
from data_generation.geom_utils import shortest_distance_idx

def gaussian(d, std, p_max):
    fx =  p_max * np.exp(-0.5 * np.square(d / std))
    return fx

def noisy_gaussian(d, std, p_max, noise_level):
    fx = gaussian(d, std, p_max)
    noise = np.random.normal(0,noise_level*fx.max(),fx.size).reshape(fx.shape)

    return fx + noise


class NeuronsSpatialFiring:

    def __init__(self, firingSettings): #Todo add gaussian shapes + max

        self.n_neurons = firingSettings["n_neurons"]
        self.std = firingSettings["std"]
        self.hyp = firingSettings["hyp"]
        self.nu_max = firingSettings["nu_max"]
        self.noise = firingSettings["noise"]
        self.fieldCenters = None
        self.firingRates = None

    def generateFiringFields(self, maze):
        """generate place fields center locations"""

        #graph hypothesis
        if self.hyp == "graph":
            self.fieldCenters = np.empty([2, self.n_neurons, maze.nb_of_trials])
            n_edges = len(maze.connectedNodes)

            #pick an edge and a percentage position on that edge + lateral distance
            firingFieldsEdges   = np.random.choice(np.arange(n_edges), self.n_neurons)
            firingFieldsPercent = np.random.choice(np.arange(100), self.n_neurons)/100
            latDelta = np.random.uniform(-0.5, 0.5, self.n_neurons)

            #place firing fields
            for i in range(maze.nb_of_trials):
                for j in range(self.n_neurons): ##TODO speed up (remove loop) if possible ?
                    nodes = maze.connectedNodes[firingFieldsEdges[j]]
                    perc = firingFieldsPercent[j]
                    lat_d = latDelta[j]
                    n2n_vec = np.asarray(maze.nodeList[i][nodes[1]])-np.asarray(maze.nodeList[i][nodes[0]])
                    ortho_vec = np.array([-n2n_vec[1], n2n_vec[0]])
                    ortho_vec = ortho_vec/(np.linalg.norm(ortho_vec))
                    self.fieldCenters[:, j, i] = maze.nodeList[i][nodes[0]] + perc*n2n_vec + lat_d*ortho_vec + np.array([0.5, 0.5])

                    if not maze.isInMaze(self.fieldCenters[:, j, i][0], self.fieldCenters[:, j, i][1], i):
                        a = 1-2/np.sqrt(2)
                        self.fieldCenters[:, j, i] = maze.nodeList[i][nodes[0]] + perc*n2n_vec + lat_d*ortho_vec*a + np.array([0.5, 0.5])

        elif self.hyp == "euclidean":
            inMazeId = np.where(maze.fullMazeFlags == True)
            rndList = random.sample(list(np.arange(0, len(inMazeId[0]))), self.n_neurons)
            self.fieldCenters = np.column_stack([inMazeId[1][rndList]/maze.mazeRes, inMazeId[0][rndList]/maze.mazeRes])
            self.fieldCenters = np.repeat(self.fieldCenters[np.newaxis, :, :], maze.nb_of_trials, axis = 0).T

        return

    def distance(self, maze, maze_config, x, centers):
        """compute distances between positions and place field centers"""

        centers_resh = np.repeat(centers[:, np.newaxis, :], x.shape[1], axis=1)

        #euclidean distances
        if self.hyp == "euclidean":
            mat = x[:, :, np.newaxis] - centers_resh
            d= np.linalg.norm(mat, axis=0)

        #graph distances
        elif self.hyp == "graph":
            cellList = np.asarray(maze.cellList[maze_config]).T+0.5

            #map positions to maze cells
            x_tiles_mapping = shortest_distance_idx(x, cellList)
            c_tiles_mapping = shortest_distance_idx(centers, cellList)

            #find shortes path between cells
            graph = build_graph(maze.edgeList[maze_config])
            cellArray = np.array(maze.cellList[maze_config])
            xCells = cellArray[x_tiles_mapping]
            cCells = cellArray[c_tiles_mapping]

            d = np.zeros([len(xCells), len(cCells)])
            for i in range(len(xCells)):
                for j in range(len(cCells)):
                    path = BFS_SP(graph, list(xCells[i]), list(cCells[j])) #shortest path
                    if path is not None:
                        path[0] = list(x[:, i])
                        path[-1] = list(centers[:, j])
                    else:
                        path = [list(x[:, i]), list(centers[:, j])]
                    d[i, j] = compute_path_distance(path)

        else:
            print("ERROR: hypothesis non-valid")
            return

        return d

    def fire(self, traj, maze): #TODO add option to chose firing fuction easily
        self.firingRates = np.empty([traj.x_traj.shape[0], self.n_neurons], )
        idx = traj.traj_cut_idx

        for i in range(sum(traj.n_traj)):
            X = np.array([traj.x_traj[idx[i]:idx[i+1]], traj.y_traj[idx[i]:idx[i+1]]])
            maze_config = traj.corr_maze_config[i]
            d = self.distance(maze, maze_config, X,  self.fieldCenters[:, :, maze_config])
            self.firingRates[idx[i]:idx[i+1], :] = noisy_gaussian(d, self.std, self.nu_max, self.noise)

        return self.firingRates



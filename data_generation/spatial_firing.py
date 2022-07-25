import numpy as np
import random
import math
from matplotlib import pyplot as plt
from data_generation.graph import build_graph, BFS_SP, compute_path_distance

def gaussian(d, std, p_max): #Todo move to utils
    K = std*np.sqrt(2*np.pi)*p_max
    #mus_resh = np.repeat(mus[:, np.newaxis, :], x.shape[1], axis=1)
    #x_resh = x

    #mat = x_resh[:, :, np.newaxis] - mus_resh
    #mat = np.empty(mus_resh.shape)


    #deltas = np.linalg.norm(mat, axis=0)

    fx =  K/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square(d / std)) #TODO replace by pmax
    return fx

def noisy_gaussian(d, std, p_max): #Todo move to utils
    fx = gaussian(d, std, p_max)
    noise = np.random.normal(0,0.001*fx.max(),fx.size).reshape(fx.shape)#TODO add noise level as param

    return fx + noise

"""def firing_proba(x, std, mus, p_max):
    K = std * np.sqrt(2 * np.pi) * p_max
    x_resh = np.moveaxis(x, 0, 2)
    x_2d = np.reshape(x_resh, [x_resh.shape[0] * x_resh.shape[1], x_resh.shape[2]])
    dist = np.linalg.norm(mus[:, None] - x_2d, axis = -1)
    spiking_prob = K / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square((dist) / std)) #TODO add ext fct for that
    spikes = np.empty(spiking_prob.shape)
    for i in range(spikes.shape[0]): ##TODO: this loop is very slow
        for j in range(spikes.shape[1]):
            spikes[i, j] = np.random.choice([0, 1], p = [1 - spiking_prob[i, j], spiking_prob[i, j]])
    return spikes"""

class NeuronsSpatialFiring:

    def __init__(self, firingSettings): #Todo add gaussian shapes + max
        self.n_neurons = firingSettings["n_neurons"]
        self.res = firingSettings["resolution"]
        self.std = firingSettings["std"]
        self.hyp = firingSettings["hyp"]

        self.fieldCenters = None
        self.firingRates = None

    def generateFiringFields(self, maze):

        if self.hyp == "graph":
            self.fieldCenters = np.empty([2, self.n_neurons, maze.nb_of_trials])

            """#get directly connected nodes (=nb of edges)
            connectedNodes = []

            graph = build_graph(maze.edgeList[0])
            keys = sorted(maze.nodeList[0].keys())
            nodeList = maze.nodeList[0]

            for j in range(len(nodeList)-1):
                for k in np.arange(j + 1, len(nodeList)):
                    path = BFS_SP(graph, nodeList[keys[j]], nodeList[keys[k]])
                    common_el = 0
                    for n in nodeList.values():
                        if (np.array(path) == n).all(axis = -1).any():
                            common_el = common_el + 1
                    if common_el == 2:
                        connectedNodes.append([keys[j], keys[k]])"""

            n_edges = len(maze.connectedNodes)

            #pick an edge and a percentage position on that edge
            firingFieldsEdges = np.random.choice(np.arange(n_edges), self.n_neurons)
            firingFieldsPercent = np.random.choice(np.arange(100), self.n_neurons)/100
            latDelta = np.random.uniform(-0.5,0.5, self.n_neurons)

            #place firing fields
            for i in range(maze.nb_of_trials):
                for j in range(self.n_neurons): ##TODO speed up (remove loop)
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

                    #self.fieldCenters[:, j, i] = maze.nodeList[i][nodes[0]] + perc*(np.asarray(maze.nodeList[i][nodes[1]]) - np.asarray(maze.nodeList[i][nodes[0]])) + np.array([0.5, 0.5])

        elif self.hyp == "euclidean":
            inMazeId = np.where(maze.fullMazeFlags == True)
            rndList = random.sample(list(np.arange(0, len(inMazeId[0]))), self.n_neurons)
            self.fieldCenters = np.column_stack([inMazeId[1][rndList]/self.res, inMazeId[0][rndList]/self.res])
            self.fieldCenters = np.repeat(self.fieldCenters[np.newaxis, :, :], maze.nb_of_trials, axis = 0).T
        return

    def distance(self, maze, maze_config, x, centers):

        centers_resh = np.repeat(centers[:, np.newaxis, :], x.shape[1], axis=1)

        if self.hyp == "euclidean":
            #compute euclidan distance
            mat = x[:, :, np.newaxis] - centers_resh
            d= np.linalg.norm(mat, axis=0)
            print("dfwd")

        elif self.hyp == "graph":
            #compute graph distance

            cellList = np.asarray(maze.cellList[maze_config]).T+0.5

            #get tiles including traj and place fields positions
            cellList_resh = np.repeat(cellList[:, :, np.newaxis], x.shape[1], axis = 2)
            mat = x[:, np.newaxis, :] - cellList_resh ##tODO function for that
            d_to_tiles = np.linalg.norm(mat, axis=0)
            x_tiles_mapping = np.argmin(d_to_tiles, axis = 0)

            cellList_resh = np.repeat(cellList[:, :, np.newaxis], centers.shape[1], axis = 2)
            mat = centers[:, np.newaxis, :] - cellList_resh  ##tODO function for that
            d_to_tiles = np.linalg.norm(mat, axis=0)
            c_tiles_mapping = np.argmin(d_to_tiles, axis=0)

            graph = build_graph(maze.edgeList[maze_config])

            cellArray = np.array(maze.cellList[maze_config])
            xCells = cellArray[x_tiles_mapping]
            cCells = cellArray[c_tiles_mapping]

            paths = []
            d = np.zeros([len(xCells), len(cCells)])
            for i in range(len(xCells)):
                for j in range(len(cCells)):
                    path = BFS_SP(graph, list(xCells[i]), list(cCells[j]))
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
            self.firingRates[idx[i]:idx[i+1], :] = noisy_gaussian(d, self.std, 0.99)
        #spikes = firing_proba(traj, self.std, self.fieldCenters, p_max = 0.99)

        return self.firingRates



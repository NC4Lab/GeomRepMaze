import numpy as np
import random
from matplotlib import pyplot as plt
from data_generation.graph import build_graph, BFS_SP

def gaussian(x, std, mus, p_max): #Todo move to utils
    K = std*np.sqrt(2*np.pi)*p_max
    mus_resh = np.repeat(mus[:, np.newaxis, :], x.shape[1], axis=1)
    x_resh = x

    mat = x_resh[:, :, np.newaxis] - mus_resh
    #mat = np.empty(mus_resh.shape)
    """for i in range(mat.shape[1]):
        for j in range(mat.shape[2]):
            mat[:, i, j] = x_resh[:, i] - mus_resh[:, i, j]"""

    deltas = np.linalg.norm(mat, axis=0)

    fx =  K/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square(deltas / std)) #TODO replace by pmax
    return fx

def noisy_gaussian(x, std, mus, p_max): #Todo move to utils
    fx = gaussian(x, std, mus, p_max)
    noise = np.random.normal(0,0.001*fx.max(),fx.size).reshape(fx.shape)#TODO add noise level as param

    return fx + noise

def firing_proba(x, std, mus, p_max):
    K = std * np.sqrt(2 * np.pi) * p_max
    x_resh = np.moveaxis(x, 0, 2)
    x_2d = np.reshape(x_resh, [x_resh.shape[0] * x_resh.shape[1], x_resh.shape[2]])
    dist = np.linalg.norm(mus[:, None] - x_2d, axis = -1)
    spiking_prob = K / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square((dist) / std)) #TODO add ext fct for that
    spikes = np.empty(spiking_prob.shape)
    for i in range(spikes.shape[0]): ##TODO: this loop is very slow
        for j in range(spikes.shape[1]):
            spikes[i, j] = np.random.choice([0, 1], p = [1 - spiking_prob[i, j], spiking_prob[i, j]])
    return spikes

class NeuronsSpatialFiring:

    def __init__(self, firingSettings): #Todo add gaussian shapes + max
        self.n_neurons = firingSettings["n_neurons"]
        self.res = firingSettings["resolution"]
        self.std = firingSettings["std"]
        self.hyp = firingSettings["hyp"]

        self.fieldCenters = None

    def generateFiringFields(self, maze):

        if self.hyp == "graph":
            self.fieldCenters = np.empty([2, self.n_neurons, maze.nb_of_trials])

            #get directly connected nodes (=nb of edges)
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
                        connectedNodes.append([keys[j], keys[k]])

            n_edges = len(connectedNodes)

            #pick an edge and a percentage position on that edge
            firingFieldsEdges = np.random.choice(np.arange(n_edges), self.n_neurons)
            firingFieldsPercent = np.random.choice(np.arange(100), self.n_neurons)/100
            latDelta = np.random.uniform(-0.5,0.5, self.n_neurons)
            for i in range(maze.nb_of_trials):
                for j in range(self.n_neurons): ##TODO speed up (remove loop)
                    nodes = connectedNodes[firingFieldsEdges[j]]
                    perc = firingFieldsPercent[j]
                    lat_d = latDelta[j]
                    n2n_vec = np.asarray(maze.nodeList[i][nodes[1]])-np.asarray(maze.nodeList[i][nodes[0]])
                    ortho_vec = np.array([-n2n_vec[1], n2n_vec[0]])
                    ortho_vec = ortho_vec/(np.linalg.norm(ortho_vec))
                    self.fieldCenters[:, j, i] = maze.nodeList[i][nodes[0]] + perc*n2n_vec + lat_d*ortho_vec + np.array([0.5, 0.5])

                    #self.fieldCenters[:, j, i] = maze.nodeList[i][nodes[0]] + perc*(np.asarray(maze.nodeList[i][nodes[1]]) - np.asarray(maze.nodeList[i][nodes[0]])) + np.array([0.5, 0.5])

        elif self.hyp == "euclidean":
            inMazeId = np.where(maze.fullMazeFlags == True)
            rndList = random.sample(list(np.arange(0, len(inMazeId[0]))), self.n_neurons)
            self.fieldCenters = np.column_stack([inMazeId[1][rndList]/self.res, inMazeId[0][rndList]/self.res])
            self.fieldCenters = np.repeat(self.fieldCenters[np.newaxis, :, :], maze.nb_of_trials, axis = 0).T
        return

    def fire(self, traj, maze): #TODO add option to chose firing fuction easily
        firing_rates = np.empty([traj.shape[1], self.n_neurons, maze.nb_of_trials])
        for i in range(maze.nb_of_trials):
            firing_rates[:, :, i] = noisy_gaussian(traj, self.std, self.fieldCenters[:, :, i], 0.99)
        #spikes = firing_proba(traj, self.std, self.fieldCenters, p_max = 0.99)

        return firing_rates







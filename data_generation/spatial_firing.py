import numpy as np
import random
from data_generation.utils.geom_utils import graph_distance, euclidean_distance

def gaussian(d, std, p_max):
    fx = p_max * np.exp(-0.5 * np.square(d / std))
    return fx

class PlaceCell:

    def __init__(self, firingSettings):

        self.n_neurons = firingSettings["n_neurons"]
        self.std = firingSettings["std"]
        self.hyp = firingSettings["hyp"]
        self.nu_max = firingSettings["nu_max"]
        self.rate_noise = firingSettings["noise_on_rate"]
        self.field_noise = firingSettings["noise_on_field"]
        self.fieldCenters = None
        self.firingRates = None
        self.noNoiseFirinRates = None

    def generateFieldCenters(self, maze):
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
                for j in range(self.n_neurons): ##TODO speed up (remove loop) ?
                    nodes = maze.connectedNodes[firingFieldsEdges[j]]
                    perc = firingFieldsPercent[j]
                    lat_d = latDelta[j]
                    n2n_vec = np.asarray(maze.nodeList[i][nodes[1]])-np.asarray(maze.nodeList[i][nodes[0]]) #node2node vector
                    ortho_vec = np.array([-n2n_vec[1], n2n_vec[0]])
                    ortho_vec = ortho_vec/(np.linalg.norm(ortho_vec))
                    self.fieldCenters[:, j, i] = maze.nodeList[i][nodes[0]] + perc*n2n_vec + lat_d*ortho_vec + np.array([0.5, 0.5])

                    if not maze.isInMaze(self.fieldCenters[:, j, i][0], self.fieldCenters[:, j, i][1], i): #remapping
                        a = -(1-2/np.sqrt(2))
                        self.fieldCenters[:, j, i] = maze.nodeList[i][nodes[0]] + perc*n2n_vec + lat_d*ortho_vec*a + np.array([0.5, 0.5])

        elif self.hyp == "euclidean":
            inMazeId = np.where(maze.fullMazeFlags == True)
            rndList = random.sample(list(np.arange(0, len(inMazeId[0]))), self.n_neurons)
            self.fieldCenters = np.column_stack([inMazeId[1][rndList]/maze.mazeRes, inMazeId[0][rndList]/maze.mazeRes])
            self.fieldCenters = np.repeat(self.fieldCenters[np.newaxis, :, :], maze.nb_of_trials, axis = 0).T

        #additive noise
        noise = np.random.uniform(-self.field_noise, self.field_noise, self.fieldCenters.shape)
        self.fieldCenters = self.fieldCenters + noise

        return

    def generate_firing_rates(self, traj, maze):
        self.firingRates = np.empty([traj.x_traj.shape[0], self.n_neurons])
        self.noNoiseFirinRates =  np.empty([traj.x_traj.shape[0], self.n_neurons])
        idx = traj.traj_cut_idx

        for i in range(len(traj.traj_cut_idx)-1): #for each trajectory
            print(str(i) + "/" + str(len(traj.traj_cut_idx)-1))
            X = np.array([traj.x_traj[idx[i]:idx[i+1]], traj.y_traj[idx[i]:idx[i+1]]])
            maze_config = traj.corr_maze_config[i]

            #firing rates without noise
            self.noNoiseFirinRates[idx[i]:idx[i + 1], :] = self.fire(maze, maze_config, X)

            #noisy firing rates
            nu = self.noNoiseFirinRates[idx[i]:idx[i + 1], :]
            self.firingRates[idx[i]:idx[i + 1], :] = nu + np.random.normal(0,self.rate_noise*self.nu_max, nu.size).reshape(nu.shape)

        return self.firingRates

    def fire(self, maze, maze_config, X):

        if self.hyp == "graph":
            d = graph_distance(maze, maze_config, X, self.fieldCenters[:, :, maze_config])
        elif self.hyp == "euclidean":
            d = euclidean_distance(X, self.fieldCenters[:, :, maze_config])

        nu = gaussian(d, self.std, self.nu_max)

        return nu

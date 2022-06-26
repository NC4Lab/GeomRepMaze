import numpy as np
import random
from matplotlib import pyplot as plt
from maze import Maze

def gaussian(x, std, mus):
    deltas = np.linalg.norm(x[:, None] - mus, axis=-1)
    return 1/(std*np.sqrt(2*np.pi))*np.exp(-0.5*np.square((deltas)/std))


class NeuronsSpatialFiring:

    def __init__(self, D, BinaryMaze = None, N=100, std = 1, mu = 1):
        self.n_neurons = N
        self.binMaze = np.asarray(BinaryMaze)
        self.res = D
        self.std = std
        self.mu = mu

        self.fieldCenters = None

    #def generate_spatial_data(self):

    def generateFiringFields(self):
        highResMaze = np.repeat(self.binMaze, 10, axis = 1)
        highResMaze = np.repeat(highResMaze, 10, axis = 0) #Todo change this, add D
        inMazeId = np.where(highResMaze == 1)
        rndList = random.sample(list(np.arange( 0, len(inMazeId[0]))), self.n_neurons)
        self.fieldCenters = np.column_stack([inMazeId[0][rndList]/self.res, inMazeId[1][rndList]/self.res])

    def fire(self, traj):

        firing_rates = gaussian(traj, self.std, self.fieldCenters)
        fig, axs = plt.subplots(self.n_neurons)
        for i in range(self.n_neurons):
            axs[i].plot(firing_rates[:, i])
        plt.show()






import numpy as np
import random
from matplotlib import pyplot as plt
from maze import Maze

def gaussian(x, std, mus):
    deltas = np.linalg.norm(x[:, None] - mus, axis=-1)
    return 1/(std*np.sqrt(2*np.pi))*np.exp(-0.5*np.square((deltas)/std))


class NeuronsSpatialFiring:

    def __init__(self, disc = 100, N=100, std = 1): #Todo add gaussian shapes + max
        self.n_neurons = N
        self.res = disc
        self.std = std

        self.fieldCenters = None

    def generateFiringFields(self, binMaze):
         #binMaze = binMaze[::-1]
        inMazeId = np.where(binMaze == True)
        rndList = random.sample(list(np.arange(0, len(inMazeId[0]))), self.n_neurons)
        self.fieldCenters = np.column_stack([inMazeId[1][rndList]/self.res, inMazeId[0][rndList]/self.res])

    def fire(self, traj): #Todo add noise

        firing_rates = gaussian(traj, self.std, self.fieldCenters)
        return firing_rates







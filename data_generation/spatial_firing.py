import numpy as np
import random
from matplotlib import pyplot as plt

def gaussian(x, std, mus, p_max): #Todo move to utils
    K = std*np.sqrt(2*np.pi)*p_max
    mus_resh = np.repeat(mus[:, np.newaxis, :], x.shape[2], axis=1)
    x_resh = np.moveaxis(x, 0, 2)
    mat = np.empty([x_resh.shape[0], mus_resh.shape[0], x_resh.shape[1], x_resh.shape[2]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i, j, :, :] = x_resh[i, :, :] - mus_resh[j, :, :]

    deltas = np.linalg.norm(mat, axis=-1)

    fx = K / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square((deltas) / std))
    return fx

def noisy_gaussian(x, std, mus, p_max): #Todo move to utils
    fx = gaussian(x, std, mus, p_max)
    noise = np.random.normal(0,0.05*fx.max(),fx.size).reshape(fx.shape)

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

    def generateFiringFields(self, binMaze):
        inMazeId = np.where(binMaze == True)
        rndList = random.sample(list(np.arange(0, len(inMazeId[0]))), self.n_neurons)
        self.fieldCenters = np.column_stack([inMazeId[1][rndList]/self.res, inMazeId[0][rndList]/self.res])

    def fire(self, traj): #TODO add option to chose firing fuction easily
        firing_rates = noisy_gaussian(traj, self.std, self.fieldCenters, 0.99)
        #spikes = firing_proba(traj, self.std, self.fieldCenters, p_max = 0.99)

        return firing_rates







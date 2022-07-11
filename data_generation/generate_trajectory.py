import numpy as np
import random
from data_generation.maze import create_octogon_from_point
from matplotlib.path import Path


class Trajectory:
    def __init__(self, trajSettings):
        self.n_steps = trajSettings["n_steps"]
        self.n_traj = trajSettings["n_traj"]

        self.x_traj = np.zeros([self.n_steps, self.n_traj])
        self.y_traj = np.zeros([self.n_steps, self.n_traj])
        self.d = trajSettings["step_size"]


    def generate_random_walk(self, maze):
        home = maze.home
        # TODO  move flag computation to maze class
        home_pol = create_octogon_from_point(home)
        home_pol = Path(np.asarray(home_pol.exterior.xy).T)

        # create binary flags for each pixels (either in or out maze tiles)
        M = maze.N * maze.mazeRes  # TODO add a fct to transform ploygon ti binary flags
        xv, yv = np.meshgrid(np.linspace(0, maze.N, M), np.linspace(0, maze.N, M))
        homeFlags = home_pol.contains_points(
            np.hstack((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis]))).reshape((M, M))

        # initial point
        for t in range(self.n_traj):

            if home is not None:
                y_start, x_start = np.where(homeFlags == 1)
                idx = np.random.choice(np.arange(len(x_start)), 1)
                self.x_traj[0, t] = x_start[idx]/ maze.mazeRes
                self.y_traj[0, t] = y_start[idx]/ maze.mazeRes

            else:
                y_start, x_start = np.where(maze.mazeFlags == 1)
                idx = np.random.choice(np.arange(len(x_start)), 1)
                self.x_traj[0, t] = x_start[idx]/maze.mazeRes
                self.y_traj[0, t] = y_start[idx]/maze.mazeRes
            #generate random trajectory with constant displacement (cst speed)
            i = 1
            while i < self.n_steps:
                validNextX, validNextY = maze.get_adjacent_points(self.x_traj[i-1, t], self.y_traj[i-1, t], self.d)
                validNext = np.column_stack([validNextX, validNextY])
                next = random.choice(validNext)
                self.x_traj[i, t] = next[0]
                self.y_traj[i, t] = next[1]
                i = i + 1

        return self.x_traj, self.y_traj





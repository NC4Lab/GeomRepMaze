import numpy as np
import random
from matplotlib import pyplot as plt


class Trajectory:
    def __init__(self, n_steps, step_size):

        self.n_steps = n_steps
        self.x_traj = np.zeros(n_steps)
        self.y_traj = np.zeros(n_steps)
        self.d = step_size


    def generate_random_walk(self, maze):

        # initial point
        #TODO add restrictions if home
        self.x_traj[0] = random.choice(np.arange(0, maze.N, self.d))
        self.y_traj[0] = random.choice(np.arange(0, maze.N, self.d))
        while not maze.isInMaze(self.x_traj[0], self.y_traj[0]): #Todo remove that loop
            self.x_traj[0] = random.choice(np.arange(0, maze.N, self.d))
            self.y_traj[0] = random.choice(np.arange(0, maze.N, self.d))

        #generate random trajectory with constant displacement (cst speed)
        i = 1
        while i < self.n_steps:
            validNextX, validNextY = maze.get_adjacent_cells(self.x_traj[i-1], self.y_traj[i-1], self.d)
            validNext = np.column_stack([validNextX, validNextY])
            next = random.choice(validNext)
            self.x_traj[i] = next[0]
            self.y_traj[i] = next[1]
            i = i + 1





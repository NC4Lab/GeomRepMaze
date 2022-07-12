import numpy as np
import random
from data_generation.maze import create_octogon_from_point
from data_generation.graph import build_graph, BFS_SP
from matplotlib.path import Path


class Trajectory:
    def __init__(self, trajSettings):
        self.n_steps = trajSettings["n_steps"]
        self.n_traj = trajSettings["n_traj"]
        self.traj_type = trajSettings["type"]

        self.x_traj = None
        self.y_traj = None
        self.d = trajSettings["step_size"]
        self.homes = trajSettings["homes"]
        self.goals = trajSettings["goals"]

    def generate_trajectory(self, maze):
        if self.traj_type == "random_walk":
            self.generate_random_walk(maze)

        elif self.traj_type == "point_to_point":
            self.generate_p2p_trajectory(maze)

        else:
            print("Error: trajectory type not valid")


    def generate_p2p_trajectory(self, maze):
        p_drift = 0.15
        self.x_traj = [] #np.empty(n_traj)#np.zeros([self.n_traj, 10000])
        self.y_traj = [] #np.zeros([self.n_traj, 10000])
        graph = build_graph(maze.edgeList[maze.trial])
        start = self.homes[maze.trial]
        goal  = self.goals[maze.trial][0]

        path = np.asarray(BFS_SP(graph, start, goal))
        path = np.add(path, 0.5)

        for i in range(self.n_traj):
            #self.x_traj[0, i] = path[0, 0] + 0.5
            #self.y_traj[0, i] = path[0, 1] + 0.5
            x = [path[0, 0]]
            y = [path[0, 1]]
            k = 0
            for j in range(len(path)-1):
                while np.linalg.norm(path[j + 1] - [x[-1], y[-1]]) > 0.1:
                    validNextX, validNextY = maze.get_adjacent_points(x[k], y[k],
                                                                      self.d)
                    validNext = np.column_stack([validNextX, validNextY])

                    vToGoal = path[j+1] - np.array([x[k], y[k]])
                    vToGoal = vToGoal/np.linalg.norm(vToGoal)*self.d
                    driftPos = np.array([x[k], y[k]]) + vToGoal

                    if maze.isInMaze(x[k] + vToGoal[0], y[k] + vToGoal[1]):
                        validNext = np.row_stack([validNext, driftPos])
                        prob = np.ones(len(validNext))*((1-p_drift)/(len(validNext)-1))
                        prob[-1] = p_drift
                        idxNext = np.random.choice(np.arange(len(validNext)), p = prob)
                    else:
                        idxNext = random.choice(np.arange(len(validNext)))

                    nextPos = validNext[idxNext]
                    x.append(nextPos[0])
                    y.append(nextPos[1])
                    k = k+1
                    """self.x_traj[i, t] = next[0]
                    self.y_traj[i, t] = next[1]"""
            self.x_traj.append(x)
            self.y_traj.append(y)

        return


    def generate_random_walk(self, maze):
        self.x_traj = np.zeros([self.n_steps, self.n_traj])
        self.y_traj = np.zeros([self.n_steps, self.n_traj])
        home = self.homes[maze.trial]
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
                y_start, x_start = np.where(maze.trialMazeFlags == 1)
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

        return





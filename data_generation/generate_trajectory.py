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
        self.traj_cut_idx = np.zeros(sum(self.n_traj)+1, dtype=int)
        self.corr_maze_config = np.zeros(sum(self.n_traj), dtype=int) #contains the maze config (trial) for each trajectory
        self.d = trajSettings["step_size"]
        self.p_drift = trajSettings["p_drift"]
        self.angular_res = trajSettings["angular_res"]
        self.speed_var = trajSettings["speed_variability"]

    def generate_trajectory(self, maze):
        if self.traj_type == "random_walk":
            self.generate_random_walk(maze)

        elif self.traj_type == "point_to_point":
            self.generate_p2p_trajectory(maze)

        else:
            print("Error: trajectory type not valid")


    def generate_p2p_trajectory(self, maze):
        self.x_traj = []
        self.y_traj = []

        #extract maze configurations for each trajectory
        self.corr_maze_config[0:self.n_traj[0]] = 0
        for i in np.arange(maze.nb_of_trials - 1) + 1:
            self.corr_maze_config[sum(self.n_traj[0:i]):sum(self.n_traj[0:i + 1])] = i

        k = 0
        for n in range(maze.nb_of_trials):
            #get path from home to goal
            graph = build_graph(maze.edgeList[n])
            start = maze.homes[n]
            goal = maze.goals[n][0]

            path = np.asarray(BFS_SP(graph, start, goal))
            path = np.add(path, 0.5)

            for i in range(self.n_traj[n]):
                #starting position
                self.x_traj.append(path[0, 0])
                self.y_traj.append(path[0, 1])

                #store index where trajectory starts
                if n == 0:
                    self.traj_cut_idx[i] = int(k)
                else:
                    self.traj_cut_idx[n * self.n_traj[n - 1] + i] = int(k)
                k = k+1

                #generate trajectory
                for j in range(len(path)-1):
                    while np.linalg.norm(path[j + 1] - [self.x_traj[-1], self.y_traj[-1]]) > 0.1: #TODO 0.1 as param
                        var = random.uniform(-self.speed_var, self.speed_var)
                        d = self.d + var*self.d

                        validNextX, validNextY = maze.get_adjacent_points(self.x_traj[k-1], self.y_traj[k-1],
                                                                          d, self.angular_res, trial = n)
                        validNext = np.column_stack([validNextX, validNextY])

                        vToGoal = path[j+1] - np.array([self.x_traj[k-1], self.y_traj[k-1]]) #vector from current pos to goal
                        vToGoal = vToGoal/np.linalg.norm(vToGoal)*d
                        driftPos = np.array([self.x_traj[k-1], self.y_traj[k-1]]) + vToGoal

                        #randomly choose next position among possible adjacent positions
                        if maze.isInMaze(self.x_traj[k-1] + vToGoal[0], self.y_traj[k-1] + vToGoal[1], trial=n):
                            validNext = np.row_stack([validNext, driftPos])
                            prob = np.ones(len(validNext))*((1-self.p_drift)/(len(validNext)-1))
                            prob[-1] = self.p_drift
                            idxNext = np.random.choice(np.arange(len(validNext)), p = prob)
                        else:
                            idxNext = random.choice(np.arange(len(validNext)))

                        nextPos = validNext[idxNext]
                        self.x_traj.append(nextPos[0])
                        self.y_traj.append(nextPos[1])
                        k = k+1

        self.x_traj = np.asarray(self.x_traj).T
        self.y_traj = np.asarray(self.y_traj).T
        self.traj_cut_idx[-1] = len(self.x_traj)

        return


    def generate_random_walk(self, maze):
        self.x_traj = np.zeros(self.n_steps * sum(self.n_traj))
        self.y_traj = np.zeros(self.n_steps * sum(self.n_traj))

        #extract maze configuration id for each trajectory
        self.corr_maze_config[0:self.n_traj[0]] = 0
        for i in np.arange(maze.nb_of_trials-1)+1:
            self.corr_maze_config[sum(self.n_traj[0:i]):sum(self.n_traj[0:i+1])] = i

        self.traj_cut_idx = np.linspace(0, self.n_steps * sum(self.n_traj), sum(self.n_traj)+1, dtype=int)
        k = 0
        for n in range(maze.nb_of_trials):
            home = maze.homes[n]
            homeFlags = maze.get_cell_flags(home)

            for t in range(self.n_traj[n]):

                # starting point
                if home is not None:
                    startField = homeFlags
                else:
                    startField = maze.trialMazeFlags[n, :, :]

                y_start, x_start = np.where(startField == 1)
                idx = np.random.choice(np.arange(len(x_start)), 1)
                self.x_traj[k] = x_start[idx] / maze.mazeRes
                self.y_traj[k] = y_start[idx] / maze.mazeRes

                k = k+1
                i = 1

                #generate random trajectory with constant displacement (rnd walk)
                while i < self.n_steps:
                    var = random.uniform(-self.speed_var, self.speed_var)
                    d = self.d + var*self.d
                    validNextX, validNextY = maze.get_adjacent_points(self.x_traj[k-1], self.y_traj[k-1], d, self.angular_res, trial = n)
                    validNext = np.column_stack([validNextX, validNextY])
                    next = random.choice(validNext)
                    self.x_traj[k] = next[0]
                    self.y_traj[k] = next[1]
                    k = k+1
                    i = i+1

        return





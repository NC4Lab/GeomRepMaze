"""GENERATE BEHAVIORAL AND NEURAL DATA"""
import numpy as np
import pickle
import os
from datetime import datetime
from data_generation.generate_trajectory import Trajectory
from data_generation.maze import Maze
from data_generation.spatial_firing import PlaceCell
from settings.custom_settings import CustomSettings
from settings.default_settings import DefaultSettings
from data_generation.plots import *

USE_CUSTOM_SETTINGS = True
PLOT = True
SAVE = True

def save_data():
    """Saves the generated data"""

    print("saving data")
    i = 0

    #check path
    while os.path.exists('./data_generation/generated_data/experiment%s' % i):
        i += 1
    os.mkdir('./data_generation/generated_data/experiment%s' % i)

    #summary output
    lines = [datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
             "hypothesis: " + placeCells.hyp,
             "trajectory type: " + traj.traj_type,
             "number of trajectories: " + str(traj.n_traj),
             "number of maze config: " + str(maze.nb_of_trials),
             "number of neurons: " + str(placeCells.n_neurons),
             "drift: " + str(traj.p_drift),
             "FR noise: " + str(placeCells.rate_noise)]

    with open('./data_generation/generated_data/experiment%s/summary.txt' % i, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    #save data structures
    with open('./data_generation/generated_data/experiment%s/trajectory.pkl' % i, 'wb') as outp:
        pickle.dump(traj, outp, pickle.HIGHEST_PROTOCOL)
    with open('./data_generation/generated_data/experiment%s/placeCells.pkl' % i, 'wb') as outp:
        pickle.dump(placeCells, outp, pickle.HIGHEST_PROTOCOL)
    with open('./data_generation/generated_data/experiment%s/maze.pkl' % i, 'wb') as outp:
        pickle.dump(maze, outp, pickle.HIGHEST_PROTOCOL)

def plot_data(maze, traj, placeCells):

    #plot an example trajectory in each maze configuartion
    idx = 0
    nb_goals = 3
    for i in range(maze.nb_of_trials):
        traj_x_array = traj.x_traj[traj.traj_cut_idx[idx]:traj.traj_cut_idx[idx + 1]]
        traj_y_array = traj.y_traj[traj.traj_cut_idx[idx]:traj.traj_cut_idx[idx + 1]]
        plot_traj_cells_maze(traj_x_array, traj_y_array, maze, placeCells, title="example of a trajectory", config=i)
        idx += nb_goals*traj.n_traj[i]

    # plot firing rates of neuron for an example trajectory
    plt.figure()
    plt.title("Firing rates of the neurons, trajectory 1")
    plt.imshow(placeCells.firingRates[0:traj.traj_cut_idx[1], :].T, interpolation='nearest', aspect='auto')
    plt.xlabel("time step")
    plt.ylabel("Neuron #")
    plt.show()

    # plot firing rate time serie of an example neuron for an example trajectory
    plt.figure()
    plt.title("Firing rates of 1 neuron")
    L = len(placeCells.firingRates[0:traj.traj_cut_idx[1], 0])
    plt.plot(np.arange(L), placeCells.firingRates[0:traj.traj_cut_idx[1], 0])
    plt.show()

def generate_data(s):

    #create objects
    maze = Maze(s.mazeSettings)
    traj = Trajectory(s.trajectorySettings)
    placeCells = PlaceCell(s.firingSettings)
    print("create maze")
    maze.createTrialMaze(s.mazeSettings)
    print("place firing fields")
    placeCells.generateFieldCenters(maze)
    print("generate trajectories")
    traj.generate_trajectory(maze)
    print("generate firing rates")
    placeCells.generate_firing_rates(traj, maze)

    return maze, traj, placeCells


############################# Main ##########################################

if __name__ == '__main__':
    print("START")

    if USE_CUSTOM_SETTINGS:
        s = CustomSettings()
    else:
        s = DefaultSettings()

    maze, traj, placeCells = generate_data(s)

    if SAVE:
        save_data()

    if PLOT:
        plot_data(maze, traj, placeCells)





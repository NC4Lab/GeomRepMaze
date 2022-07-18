import numpy as np
import pickle
import os
from datetime import datetime
from matplotlib import pyplot as plt
from data_generation.generate_trajectory import Trajectory
from data_generation.maze import Maze
from data_generation.spatial_firing import NeuronsSpatialFiring
from settings.custom_settings import CustomSettings
from settings.default_settings import DefaultSettings

USE_CUSTOM_SETTINGS = 1
PLOT = 1

############################# Main ##########################################
if __name__ == '__main__':
    print("START")

    if USE_CUSTOM_SETTINGS:
        s = CustomSettings()
    else:
        s = DefaultSettings()

    #create objects
    maze = Maze(s.mazeSettings)
    traj = Trajectory(s.trajectorySettings)
    placeCells = NeuronsSpatialFiring(s.firingSettings)

    print("create maze")
    maze.createTrialMaze(s.mazeSettings)
    print("place firing fields")
    placeCells.generateFiringFields(maze)
    print("generate trajectories")
    traj.generate_trajectory(maze)
    print("generate firing rates")
    firingRates = placeCells.fire(np.array([traj.x_traj, traj.y_traj]), maze)

    print("saving data")

    ##save data
    i = 0
    while os.path.exists('./data_generation/generated_data/experiment%s' % i):
        i += 1
    os.mkdir('./data_generation/generated_data/experiment%s' % i)

    lines = [datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
             "hypothesis: " + placeCells.hyp,
             "trajectory type: " + traj.traj_type,
             "number of maze config: " + str(maze.nb_of_trials),
             "number of neurons: " + str(placeCells.n_neurons)]

    with open('./data_generation/generated_data/experiment%s/summary.txt' %i, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


    with open('./data_generation/generated_data/experiment%s/trajectory.pkl' % i, 'wb') as outp:
        pickle.dump(traj, outp, pickle.HIGHEST_PROTOCOL)
    with open('./data_generation/generated_data/experiment%s/placeCells.pkl' % i, 'wb') as outp:
        pickle.dump(maze, outp, pickle.HIGHEST_PROTOCOL)
    with open('./data_generation/generated_data/experiment%s/maze.pkl' % i, 'wb') as outp:
        pickle.dump(maze, outp, pickle.HIGHEST_PROTOCOL)



    #####PLOTS
    if PLOT:
        for i in range(maze.nb_of_trials):
            im = plt.figure()
            plt.title("trajectory 1")
            idx = np.where(traj.corr_maze_config == i)[0][0]
            plt.plot(traj.x_traj[traj.traj_cut_idx[idx]:traj.traj_cut_idx[idx+1]], traj.y_traj[traj.traj_cut_idx[idx]:traj.traj_cut_idx[idx+1]], label = "trajectory")
            plt.plot(traj.x_traj[traj.traj_cut_idx[idx]], traj.y_traj[traj.traj_cut_idx[idx]], 'ko', label = "start")
            plt.plot(traj.x_traj[traj.traj_cut_idx[idx+1]-1], traj.y_traj[traj.traj_cut_idx[idx+1]-1], 'k*', label = "stop")
            plt.xlim([0, maze.N])
            plt.ylim([0, maze.N])
            plt.grid()
            plt.plot(placeCells.fieldCenters[0, :, i], placeCells.fieldCenters[1, :, i], 'r*', label = "place field centers")

            plt.legend()
            plt.imshow(maze.trialMazeFlags[i, ::-1], extent = [0, maze.N, 0, maze.N])
            plt.show()
        #Neuron firing fields

        plt.figure()
        plt.title("Firing rates of the neurons, trajectory 1")
        plt.imshow(firingRates[0:traj.traj_cut_idx[1], :, 0].T, interpolation='nearest', aspect='auto')
        plt.xlabel("time step")
        plt.ylabel("Neuron #")
        plt.show()

        plt.figure()
        plt.title("Firing rates of 1 neuron")
        L = len(firingRates[0:traj.traj_cut_idx[1], 0, 0])
        plt.plot(np.arange(L), firingRates[0:traj.traj_cut_idx[1], 0, 0])
        plt.show()



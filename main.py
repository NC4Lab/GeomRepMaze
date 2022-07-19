import numpy as np
from matplotlib import pyplot as plt
from data_generation.generate_trajectory import Trajectory
from data_generation.maze import Maze
from data_generation.spatial_firing import NeuronsSpatialFiring
from sklearn.model_selection import train_test_split
from models.MLP import MLP, PrepareDataset
from torch import nn
import torch
from sklearn.preprocessing import StandardScaler
from settings.custom_settings import CustomSettings
from settings.default_settings import DefaultSettings

####################################Constants##########


############################# Main ##########################################
if __name__ == '__main__':
    print("START")
    s = CustomSettings()

#   maze = Maze(s.mazeSize, octoMazeBool = s.octoMazeBool, home= s.home)
    maze = Maze(s.mazeSettings)
    traj = Trajectory(s.trajectorySettings)
    placeFields = NeuronsSpatialFiring(s.firingSettings)
    print("create maze")
    maze.createTrialMaze(s.mazeSettings)
    print("place firing fields")
    placeFields.generateFiringFields(maze)
    print("generate trajectories")
    traj.generate_trajectory(maze)
    print("generate firing rates")
    firingRates = placeFields.fire(np.array([traj.x_traj, traj.y_traj]), maze)

    ##PLOTS
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
        plt.plot(placeFields.fieldCenters[0, :, i], placeFields.fieldCenters[1, :, i], 'r*', label = "place field centers")

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


###########Train Model
    """M = s.trajectorySettings["n_steps" * s.trajectorySettings["n_traj"]]
    X = np.reshape(np.moveaxis(firingRates, 2, 0), [M, s.firingSettings["n_neurons"]])
    y = np.reshape(np.array([traj.x_traj.T, traj.y_traj.T]), [2, M]).T
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y)
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.33)
    np.save("./test_data/X_train", X_train)
    np.save("./test_data/y_train", y_train)
    np.save("./test_data/X_test", X_test)
    np.save("./test_data/y_test", y_test)"""

#    model = MLP(n_neurons, loss_fct = nn.L1Loss(), opt = torch.optim.Adam, lr = 1e-4)
#    model.run_training(X_train, y_train, nb_epochs = 15, batch_size = 10)
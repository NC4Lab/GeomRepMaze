import numpy as np
from matplotlib import pyplot as plt
from settings.compare_settings import CompareSettings
from data_generation.generate_data import generate_data
from data_generation.spatial_firing import NeuronsSpatialFiring

GraphSettings = CompareSettings()
GraphSettings.firingSettings["hyp"] = "graph"

EuclideanSettings = CompareSettings()
EuclideanSettings.firingSettings["hyp"] = "euclidean"

maze, traj, placeCells_graph = generate_data(GraphSettings)

placeCells_euclid = NeuronsSpatialFiring(EuclideanSettings.firingSettings)
placeCells_euclid.generateFiringFields(maze)
placeCells_euclid.fire(traj, maze)



#Plots
for i in range(maze.nb_of_trials):
    plt.figure()
    plt.title("Firing rates of the neurons, traj %s, graph hyp" %i)
    plt.imshow(placeCells_graph.firingRates[traj.traj_cut_idx[i]:traj.traj_cut_idx[i+1], :].T, interpolation='nearest', aspect='auto')
    plt.xlabel("time step")
    plt.ylabel("Neuron #")
    plt.show()

    plt.figure()
    plt.title("Firing rates of the neurons, traj %s, euclidean hyp" %i)
    plt.imshow(placeCells_euclid.firingRates[traj.traj_cut_idx[i]:traj.traj_cut_idx[i+1], :].T, interpolation='nearest', aspect='auto')
    plt.xlabel("time step")
    plt.ylabel("Neuron #")
    plt.show()

for i in range(maze.nb_of_trials):
    im = plt.figure()
    plt.title("trajectory %s, graph" % i)
    plt.plot(traj.x_traj[traj.traj_cut_idx[i]:traj.traj_cut_idx[i + 1]],
             traj.y_traj[traj.traj_cut_idx[i]:traj.traj_cut_idx[i + 1]], label="trajectory")
    plt.plot(traj.x_traj[traj.traj_cut_idx[i]], traj.y_traj[traj.traj_cut_idx[i]], 'ko', label="start")
    plt.plot(traj.x_traj[traj.traj_cut_idx[i+1] - 1], traj.y_traj[traj.traj_cut_idx[i+1] - 1], 'kx',
             label="stop")
    plt.xlim([0, maze.N])
    plt.ylim([0, maze.N])
    plt.grid()
    plt.scatter(placeCells_graph.fieldCenters[0, :, i], placeCells_graph.fieldCenters[1, :, i], marker='*', c='r',
                label="place field centers")

    plt.legend()
    plt.imshow(maze.trialMazeFlags[i, ::-1], cmap='Greens', extent=[0, maze.N, 0, maze.N])
    plt.show()

for i in range(maze.nb_of_trials):
    im = plt.figure()
    plt.title("trajectory %s, euclid" % i)
    idx = np.where(traj.corr_maze_config == i)[0][0]
    plt.plot(traj.x_traj[traj.traj_cut_idx[idx]:traj.traj_cut_idx[idx + 1]],
             traj.y_traj[traj.traj_cut_idx[idx]:traj.traj_cut_idx[idx + 1]], label="trajectory")
    plt.plot(traj.x_traj[traj.traj_cut_idx[idx]], traj.y_traj[traj.traj_cut_idx[idx]], 'ko', label="start")
    plt.plot(traj.x_traj[traj.traj_cut_idx[idx + 1] - 1], traj.y_traj[traj.traj_cut_idx[idx + 1] - 1], 'kx',
             label="stop")
    plt.xlim([0, maze.N])
    plt.ylim([0, maze.N])
    plt.grid()
    plt.scatter(placeCells_euclid.fieldCenters[0, :, i], placeCells_euclid.fieldCenters[1, :, i], marker='*',
                c='r', label="place field centers")

    plt.legend()
    plt.imshow(maze.trialMazeFlags[i, ::-1], cmap='Greens', extent=[0, maze.N, 0, maze.N])
    plt.show()

import numpy as np
from matplotlib import pyplot as plt

from generate_trajectory import Trajectory
from maze import Maze
from spatial_firing import NeuronsSpatialFiring


##################################### Variables ###############################

#Maze
octoMazeBool = True
mazeSize = 7
mazeCells = list([[[2, 0], [2, 1], [1, 2], [0, 3]],
                     [[2, 0], [2, 1], [3, 2], [4, 3]],
                     [[2, 0], [2, 1], [3, 1], [4, 1], [5, 1]]])
Home = [2, 0]
G1 = [0, 3]
G2 = [4, 3]
G3 = [5, 1]

#Neurons
n_neurons = 4
disc = 100 #discretization of each maze cells to chose firing field center position

#Trajectory

#rnd walk
n_steps = 1000
step_size = 1/10

############################# Main ##########################################
print("START")
maze = Maze(mazeSize, octoMazeBool = octoMazeBool)
traj = Trajectory(n_steps, step_size)
placeFields = NeuronsSpatialFiring(disc = disc, N = n_neurons)

print("create maze")
maze.create_maze(mazeCells)

print("place firing fields")
placeFields.generateFiringFields(maze.mazeFlags)

print("generate a trajectory")
traj.generate_random_walk(maze)

print("generate firing rates")
firingRates = placeFields.fire(np.column_stack([traj.x_traj, traj.y_traj]))


##PLOTS
im = plt.figure()
plt.title("Random Walk in maze")
"""c = np.tan(np.linspace(-1, 1, len(traj.x_traj)))
plt.scatter(traj.x_traj, traj.y_traj, c = c)"""
plt.plot(traj.x_traj, traj.y_traj)

plt.xlim([0, maze.N])
plt.ylim([0, maze.N])
plt.grid()
plt.imshow(maze.mazeFlags[::-1], extent = [0, maze.N, 0, maze.N])

#Neuron firing fields
plt.plot(placeFields.fieldCenters[:, 0], placeFields.fieldCenters[:, 1], 'r*')
plt.show()

fig, axs = plt.subplots(n_neurons)
for i in range(n_neurons):
    axs[i].plot(firingRates[:, i])
plt.show()





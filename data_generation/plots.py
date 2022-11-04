from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import seaborn as sns

def plot_traj_cells_maze(traj_x_array, traj_y_array, maze, placeCells, title, config):
    im = plt.figure()
    plt.title(title)
    # idx = np.where(traj.corr_maze_config == i)[0][0]
    plt.plot(traj_x_array, traj_y_array, label="trajectory")
    plt.plot(traj_x_array[0], traj_y_array[0], 'ko', label="start")
    plt.plot(traj_x_array[-1], traj_y_array[-1], 'kx',
             label="stop")
    plt.xlim([0, maze.N])
    plt.ylim([0, maze.N])
    plt.grid()
    plt.scatter(placeCells.fieldCenters[0, 0, config], placeCells.fieldCenters[1, 0, config], marker='o',
                c='k', label="place field centers")
    plt.scatter(placeCells.fieldCenters[0, :, config], placeCells.fieldCenters[1, :, config], marker='*',
                c='r', label="place field centers")

    plt.legend()
    plt.imshow(maze.trialMazeFlags[config, ::-1], cmap='Greens', extent=[0, maze.N, 0, maze.N])
    plt.show()

"""def plot_traj_maze(traj_x_array, traj_y_array, maze, title, config, maze_color="Greens"):

    im = plt.figure()
    plt.title(title)
    plt.plot(traj_x_array, traj_y_array, label="trajectory")
    plt.plot(traj_x_array[0], traj_y_array[0], 'ko', label="start")
    plt.plot(traj_x_array[-1], traj_y_array[-1], 'kx',
             label="stop")
    plt.xlim([0, maze.N])
    plt.ylim([0, maze.N])
    plt.grid()
    plt.legend()
    plt.imshow(maze.trialMazeFlags[config, ::-1], cmap=maze_color, extent=[0, maze.N, 0, maze.N])
    plt.show()"""

"""def ax_plot_traj(ax, traj_x_array, traj_y_array, maze, config, maze_color="Greens"):
    #ax.title(title)
    col = sns.color_palette("colorblind")

    ax.plot(traj_x_array, traj_y_array)
    ax.plot(traj_x_array[0], traj_y_array[0], 'yo')
    ax.plot(traj_x_array[-1], traj_y_array[-1], 'yx')
    ax.set_xlim([0, maze.N])
    ax.set_ylim([0, maze.N])
    ax.grid()
    patch = patches.PathPatch(maze.trialOctoMaze[config], facecolor='None', lw=3)
    ax.add_patch(patch)
    ax.imshow(maze.trialMazeFlags[config, ::-1], cmap=maze_color, extent=[0, maze.N, 0, maze.N], alpha=0.01)"""

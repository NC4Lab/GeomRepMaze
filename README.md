# Maze_analysis
Computational analysis of the maze data (behavioral and neural data).

## Synthetic data 
The folder **data_generation/** contains the code to generate sythetic data. Run **generate_data.py** to generate sythetic data

The following paramaters can be changed at the top of the file:
- USE_CUSTOM_SETTINGS: if set to 1, custom settings are used. If set to 0, default settings are used-
- PLOT: Whether to display the synthetized data
- SAVE: Whether to save the synthetized data. The data is saved in the folder _generated_data/_

### Settings

Custom settings can be defined in **settings/custom_settings.py**:
#### maze settings
- size: width of the maze
- octogonalMaze: whether to use a octogonal or square cells
- cellList: list of cells involved in each maze configuration
- edgeList: lists of connected cells (one list per maze configuration)
- nodeList: list of node cells (one list per maze configuration)
- nb_of_trials: number of maze configuration. Shall be consistent with cellList, edgeList and nodeList.
- resolution: pixel width of each cell. The resolution is involved in computations to find out whether a point is contained in a cell. 
#### trajectory settings
- type: "point_to_point" or "random_walk"
- n_traj: list containinf the number of trajectories generated per maze configuration
- n_steps: number of steps per trajectory (only for random walk)
- step_size
- homes: list of home positions for each maze configuration
- goals: list of goal positions for each maze configuration
#### firing settings
- hyp: hypothesis. "euclidean" or "graph"
- n_neurons: number of neurons (place cells)
- std: std of the gaussian function (for firing rate computation)






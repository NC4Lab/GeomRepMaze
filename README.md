# Maze_analysis
Simulation of place cell population activities (encoding either Euclidean- or
graph-based position) and computational analysis of the generated data..

## Synthetic data 
The folder **data_generation/** contains the code to generate sythetic data. Run **generate_data.py** to generate sythetic data

The following paramaters can be changed at the top of the file:
- USE_CUSTOM_SETTINGS: if set to 1, custom settings are used. If set to 0, default settings are used-
- PLOT: Whether to display the synthetized data
- SAVE: Whether to save the synthetized data. 

### Saving 
The data is saved in the folder _generated_data/_. 

For each experiment, three class objects (corresponding to trajectory, maze and place cells) are saved as .pkl files and a summary text file is generated. The summary file contains information about the time of the experiment and the main settings.

### Settings

Custom settings can be defined in **settings/custom_settings.py**:
#### maze settings
- size: width of the maze
- octogonalMaze: whether to use a octogonal or square cells
- cellList: list of cells involved in each maze configuration
- edgeList: lists of connected cells (one list per maze configuration)
- nodeList: list of node cells (one list per maze configuration)
- nb_of_trials: number of maze configuration. Shall be consistent with cellList, edgeList and nodeList.
- homes: list of home positions for each maze configuration
- goals: list of goal positions for each maze configuration
- resolution: pixel width of each cell. The resolution is involved in computations to find out whether a point is contained in a cell. 
#### trajectory settings
- type: "point_to_point" or "random_walk"
- n_traj: list containinf the number of trajectories generated per maze configuration
- n_steps: number of steps per trajectory (only for random walk)
- step_size
- p_drift: strength of the drift towards goal positiom (between 0 and 1. 0 gives random walk, 1 gives direct trajectory to the goal node)
- angular_res: angule resolution of the trajectory, i.e. number of directions the rat can move toward. 
- speed_variabilty: variability of the step size (in percentage of _step_size_)

#### firing settings
- hyp: hypothesis. "euclidean" or "graph"
- n_neurons: number of neurons (place cells)
- std: std of the gaussian function (for firing rate computation)
- nu_max: max firing rate of the neurons
- noise_on_rate: noise level on the firing rates (in percentage of the max firing rate)
- noise_on_field: noise on place field centers' positions. For each place field center coordinate, the noise is randomly (uniform dist) chosen between -noise_on_field and +noise_on_field.






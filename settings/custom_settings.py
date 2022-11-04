
from data_generation.maze_configs import nodeList


class CustomSettings():

    def __init__(self):
        self.mazeSettings = {
            "size": 7,
            "octogonalMaze": True,
            "nodeList": nodeList,
            "nb_of_trials": 3,#len(nodeList),
            "resolution": 100
        }

        self.trajectorySettings = {
            "type": "point_to_point", #"random_walk", "point_to_point"
            "n_traj": [15, 15, 15], #[2, 2, 2, 2, 2, 2,2 , 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "n_steps": 100000, #only used for random walk
            "step_size": 1 / 100,
            "p_drift": 0.1,
            "angular_res": 360,
            "speed_variability": 0
        }

        self.firingSettings = {
            "hyp": "graph", #"euclidean", "graph"
            "n_neurons": 100,
            "std": 0.3,
            "nu_max": 40,
            "noise_on_rate": 0,
            "noise_on_field": 0
        }
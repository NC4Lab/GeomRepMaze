
mazeCellsT1 = list([[2, 0], [2, 1], [1, 2], [0, 3], [3, 2], [4, 3], [3, 1], [4, 1], [5, 1]])
mazeCellsT2 = list([[0, 3], [1, 4], [2, 5], [3, 6], [3, 5], [4, 5], [5, 5], [6, 5],
                    [3, 4], [4, 3]])
mazeCellsT3 = list([[6, 5], [5, 4], [5, 3], [5, 2], [5, 1], [4, 3], [4, 5], [3, 6]])


edgesT1 = [[[2, 0], [2, 1]],
           [[2, 1], [3, 1]],
           [[2, 1], [3, 2]],
           [[2, 1], [1, 2]],
           [[1, 2], [0, 3]],
           [[3, 1], [4, 1]],
           [[4, 1], [5, 1]],
           [[2, 1], [3, 2]],
           [[3, 2], [4, 3]]]

edgesT2 = [[[0, 3], [1, 4]],
           [[1, 4], [2, 5]],
           [[2, 5], [3, 6]],
           [[2, 5], [3, 5]],
           [[3, 5], [4, 5]],
           [[4, 5], [5, 5]],
           [[5, 5], [6, 5]],
           [[2, 5], [3, 4]],
           [[3, 4], [4, 3]]]

edgesT3 = [[[6, 5], [5, 4]],
           [[5, 4], [5, 3]],
           [[5, 3], [5, 2]],
           [[5, 2], [5, 1]],
           [[5, 4], [4, 3]],
           [[5, 4], [4, 5]],
           [[4, 5], [3, 6]]]

nodesT1 = {
    "A": [2, 0],#home
    "B": [2, 1],#intersection
    "C": [0, 3],#g1
    "D": [4, 3],#g2
    "E": [5, 1],#g3
}

nodesT2 = {
    "A": [0, 3],
    "B": [2, 5],
    "C": [3, 6],
    "D": [6, 5],
    "E": [4, 3],
}

nodesT3 = {
    "A": [6, 5],
    "B": [5, 4],
    "C": [5, 1],
    "D": [4, 3],
    "E": [3, 6],
}

class CustomSettings():

    def __init__(self):
        self.mazeSettings = {
            "size": 7,
            "octogonalMaze": True,
            "cellList": [mazeCellsT1, mazeCellsT2, mazeCellsT3],
            "edgeList": [edgesT1, edgesT2, edgesT3],
            "nodeList": [nodesT1, nodesT2, nodesT3],
            "homes": [[2, 0], [0, 3], [6, 5]],
            "goals": [[[4, 3], [4, 3], [5, 1]], [[3, 6], [6, 5], [4, 3]], [[5, 1], [4, 3], [3, 6]]],
            "nb_of_trials": 3,
            "resolution": 100
        }

        self.trajectorySettings = {
            "type": "point_to_point", #"random_walk", "point_to_point"
            "n_traj": [5, 5, 5],
            "n_steps": 1000, #only used for random walk
            "step_size": 1 / 10,
            "p_drift": 0.15,
            "angular_res": 360,
            "speed_variability": 0.5
        }

        self.firingSettings = {
            "hyp": "euclidean", #"euclidean", "graph", "reward"
            "n_neurons": 3,
            "std": 0.7,
            "nu_max": 100,
            "noise_on_rate": 0.01,
            "noise_on_field": 0.1
        }



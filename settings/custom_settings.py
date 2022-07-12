import numpy as np

mazeCellsT1 = list([[2, 0], [2, 1], [1, 2], [0, 3], [3, 2], [4, 3], [3, 1], [4, 1], [5, 1]])
mazeCellsT2 = list([[0, 3], [1, 4], [2, 5], [3, 6], [3, 5], [4, 5], [5, 5], [6, 5],
                    [3, 4], [4, 3]])

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


"""mazeCellsT1 = list([[[2, 0], [2, 1], [1, 2], [0, 3]],
                    [[2, 0], [2, 1], [3, 2], [4, 3]],
                    [[2, 0], [2, 1], [3, 1], [4, 1], [5, 1]]])
mazeCellsT2 = list([[[0, 3], [1, 4], [2, 5], [3, 6]],
                    [[0, 3], [1, 4], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5]],
                    [[0, 3], [1, 4], [2, 5], [3, 4], [4, 3]]])
mazeCellsT3 = list([[[6, 5], [5, 4], [5, 3], [5, 2], [5, 1]],
                    [[6, 5], [5, 4], [4, 3]],
                    [[6, 5], [5, 4], [4, 5], [3, 6]]])
"""

class CustomSettings():

    def __init__(self):
        self.mazeSettings = {
            "size": 7,
            "octogonalMaze": True,
            "cellList": [mazeCellsT1, mazeCellsT2],
            "edgeList": [edgesT1, edgesT2],
            "trial_nb": 0,
            "resolution": 100
        }

        self.trajectorySettings = {
            "type": "random_walk", #"random_walk", "point_to_point"
            "n_traj": 1,
            "n_steps": 1000,
            "step_size": 1 / 10,
            "homes": [[2, 0], [3, 0]],
            "goals": [[[4, 3], [4, 3], [5, 1]], [[3, 6], [6, 5], [4, 3]]]
        }

        self.firingSettings = {
            "hyp": "euclidean",
            "n_neurons": 100,
            "resolution": 100,
            "std": 0.7,
        }

        self.modelSettings = {
            "model": "MLP"
        }

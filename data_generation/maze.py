import numpy as np

"Class to define a Square Maze with obstacles"
class Maze():

    def __init__(self, N, D):
        self.N = N #Maze width
        self.disc = D #dicret. per cell
        self.M = N*D #size of the maze in disc space

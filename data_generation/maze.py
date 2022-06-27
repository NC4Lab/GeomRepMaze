import numpy as np

"Class to define a Square Maze with obstacles"
class Maze():

    def __init__(self, N):
        #initialize 2D space
        self.N = N #Maze width

        self.binaryMaze = np.zeros((self.N, self.N), dtype=bool) #contain routes (1 = route, 0 = non-route)
        self.home = None
        self.goals = None

    def createMaze(self, mazeCells, home = None, goals = None):
        #create maze routes

        self.home = home
        self.goals = goals
        self.binaryMaze[tuple(mazeCells.T)] = True #set routes

    def isInMaze(self, x, y):
        #check if point in the maze
        x_cor = int(np.floor(x))
        y_cor = int(np.floor(y))

        if x_cor == 7:
            x_cor = 6
        if y_cor == 7:
            y_cor = 6

        if y_cor < 0 or x_cor < 0 or y_cor > 7 or x_cor > 7:
            return False
        if self.binaryMaze[x_cor, y_cor] == 1:
            return True
        else:
            return False

    def get_adjacent_cells(self, x_coor, y_coor, d):
        #returns cells adjacent to a point that are in the maze

        list_adjcoorX = []
        list_adjcoorY = []

        for x, y in [(x_coor + i, y_coor + j) for i in (-d, 0, d) for j in (-d, 0, d) if i != 0 or j != 0]:
            if self.isInMaze(x, y):
                list_adjcoorX.append(x)
                list_adjcoorY.append(y)

        return np.asarray(list_adjcoorX), np.asarray(list_adjcoorY)


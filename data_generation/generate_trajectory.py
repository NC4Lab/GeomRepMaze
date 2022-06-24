import numpy as np
import random
from matplotlib import pyplot as plt

def isInMaze(x, y, mazeBinMat, D):

    x_cor = int(np.floor(x/D))
    y_cor = int(np.floor(y/D))

    if x_cor == 7:
        x_cor = 6
    if y_cor == 7:
        y_cor = 6

    if y_cor < 0  or x_cor < 0 or y_cor > 7 or x_cor > 7:
        return False
    if mazeBinMat[x_cor, y_cor] == 1:
        return True
    else:
        return False

#Parameters
N = 7 #Maze size
D = 10 #maze discretization per cell
M = N*D

MazeCells = np.array([[0,0],[0,1], [1, 1], [1, 2], [2,2], [2, 3], [3, 2], [2, 4], [3, 1], [4, 1]])
MazeBinary = np.zeros((N,N))
MazeBinary[tuple(MazeCells.T)] = 1
print(MazeBinary)
#generate random walk

# defining the number of steps
n = 10000

# creating to array for containing x and y coordinate
# of size equals to the number of size and filled up with 0's
x_traj = np.zeros(n)
y_traj = np.zeros(n)

# filling the coordinates with random variables
x_traj[0] = random.randint(0, M)
y_traj[0] = random.randint(0, M)
while not isInMaze(x_traj[0], y_traj[0], MazeBinary, D):
    x_traj[0] = random.randint(0, M)
    y_traj[0] = random.randint(0, M)

i = 0
while i < n:
    r = random.randint(0,7)
    #wall constraints
    if r == 0:
        x_traj[i] = x_traj[i - 1] + 1
        y_traj[i] = y_traj[i-1]
    elif r == 1:
        x_traj[i] = x_traj[i - 1] -1
        y_traj[i] = y_traj[i - 1]
    elif r == 2:
        x_traj[i] = x_traj[i - 1]
        y_traj[i] = y_traj[i - 1] + 1
    elif r == 3:
        x_traj[i] = x_traj[i - 1]
        y_traj[i] = y_traj[i - 1] - 1
    elif r == 4:
        x_traj[i] = x_traj[i - 1] + 1
        y_traj[i] = y_traj[i - 1] + 1
    elif r == 5:
        x_traj[i] = x_traj[i - 1] - 1
        y_traj[i] = y_traj[i - 1] - 1
    elif r == 6:
        x_traj[i] = x_traj[i - 1] + 1
        y_traj[i] = y_traj[i - 1] - 1
    elif r == 7:
        x_traj[i] = x_traj[i - 1] - 1
        y_traj[i] = y_traj[i - 1] + 1

    if isInMaze(x_traj[i], y_traj[i], MazeBinary, D):
        i = i+1




mat = np.zeros((M,M))
mat[(x_traj.astype(int)-1, y_traj.astype(int)-1)] = 1
# plotting stuff:
im = plt.figure()
plt.title("Random Walk ($n = " + str(n) + "$ steps) in maze")
plt.plot(x_traj/D, y_traj/D)
plt.xlim([0, N])
plt.ylim([0, N])
plt.grid()
plt.imshow(MazeBinary.T[::-1], extent = [0, N, 0, N])
plt.show()






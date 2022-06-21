import numpy as np
import random
from matplotlib import pyplot as plt

#Parameters
N = 7 #Maze size
D = 10 #maze discretization per cell
M = N*D

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

for i in range(1, n):
    #wall constraints
    if x_traj[i-1] -1 < 0:
        x_traj[i] = x_traj[i - 1] + random.randint(0, 1)
    elif x_traj[i-1] + 1 > M:
        x_traj[i] = x_traj[i - 1] + random.randint(-1, 0)
    else:
        x_traj[i] = x_traj[i - 1] + random.randint(-1, 1)

    if y_traj[i-1] -1 < 0:
        y_traj[i] = y_traj[i - 1] + random.randint(0, 1)
    elif y_traj[i-1] + 1 > M:
        y_traj[i] = y_traj[i - 1] + random.randint(-1, 0)
    else:
        y_traj[i] = y_traj[i - 1] + random.randint(-1, 1)


# plotting stuff:
plt.title("Random Walk ($n = " + str(n) + "$ steps) in maze")
plt.plot(x_traj/D, y_traj/D)

plt.xlim([0, N])
plt.ylim([0, N])
plt.grid()

#pylab.savefig("rand_walk" + str(n) + ".png", bbox_inches="tight", dpi=600)
plt.show()







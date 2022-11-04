import numpy as np
from matplotlib import pyplot as plt
from models.archive.MLP import MLP
from torch import nn
import torch
from sklearn.preprocessing import StandardScaler


model = MLP(100, loss_fct = nn.L1Loss(), opt = torch.optim.Adam, lr = 1e-4)
model.load_state_dict(torch.load("./saved_models/trained_mlp_low_data"))
model.eval()
y_test = np.load("test_data/y_test.npy")
X_test = np.load("test_data/X_test.npy")
y_train = np.load("test_data/y_train.npy")
X_train = np.load("test_data/X_train.npy")

X_test_scaled = StandardScaler().fit_transform(X_test)
X_train_scaled = StandardScaler().fit_transform(X_train) #Todo this is not correct, scaling needed on whole dataset
#TODO as we know bound, easier to map!

y_pred = model.forward(torch.tensor(X_test_scaled, dtype = torch.float32))
y_pred_on_training_set = model.forward(torch.tensor(X_train_scaled, dtype = torch.float32))

plt.figure()
plt.title("ground truth (test data)")
plt.scatter(y_test[1:10000, 0], y_test[1:10000, 1])
plt.show()

plt.figure()
plt.title("prediction (test data)")
plt.scatter(y_pred.detach().numpy()[1:10000, 0], y_pred.detach().numpy()[1:10000, 1])
plt.show()

plt.figure()
plt.title("ground truth VS prediction (test data)")
plt.scatter(y_test[1:10000, 0], y_test[1:10000, 1], label = "ground truth")
plt.scatter(y_pred.detach().numpy()[1:10000, 0], y_pred.detach().numpy()[1:10000, 1], label = "prediction")
plt.legend()
plt.show()


plt.figure()
plt.title("ground truth vs prediction (training data)")
plt.scatter(y_train[1:10000, 0], y_train[1:10000, 1], label = "ground truth")
plt.scatter(y_pred_on_training_set.detach().numpy()[1:10000, 0], y_pred_on_training_set.detach().numpy()[1:10000, 1], label = "prediction")
plt.legend()
plt.show()

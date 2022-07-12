import numpy as np
from matplotlib import pyplot as plt
from data_generation.generate_trajectory import Trajectory
from data_generation.maze import Maze
from data_generation.spatial_firing import NeuronsSpatialFiring

import torch
from torch import nn

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

class PrepareDataset(torch.utils.data.Dataset):
  '''
  Prepare the Boston dataset for regression
  '''

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      #if scale_data:
         # X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]


class MLP(nn.Module):
    ''' Multilayer Perceptron for regression.'''
    def __init__(self, input_size, loss_fct = nn.L1Loss(), opt = torch.optim.Adam, lr = 1e-4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.loss_function = loss_fct
        self.optimizer = opt(self.parameters(), lr=lr)

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)

    def run_training(self,X_train, y_train, nb_epochs = 5, batch_size = 1):

        train_dataset = PrepareDataset(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


        for epoch in range(0, nb_epochs):  # 5 epochs at maximum

            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):

                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 2))

                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                outputs = self(inputs)

                # Compute loss
                loss = self.loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 10 == 0:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 500))
                    current_loss = 0.0

        # Process is complete.
        print('Training process has finished.')
        torch.save(self.state_dict(), "./saved_models/trained_mlp_low_data")











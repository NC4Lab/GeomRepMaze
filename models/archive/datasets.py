import torch
import pickle
import numpy

class PlaceDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_, fold=0, samples_per_epoch=500, device='cpu', full=False):

        if not full:
            X_ = X_[..., :30, :]  # For DGM we use modality 1 (M1) for both node representation and graph learning.

        self.n_features = X_.shape[-2]
        self.X = torch.from_numpy(X_[:, :, fold]).float().to(device)
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X, [[]]
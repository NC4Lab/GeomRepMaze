import torch
import numpy as np
from torch import nn, optim
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

#Lightning module AE
class Lit_AE(LightningModule):
    def __init__(self, auto_encoder):
        super().__init__()
        self.auto_encoder = auto_encoder #takes an AE architecture as input
        self.metric = torch.nn.MSELoss()

    def forward(self, x):
        return self.auto_encoder.forward(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.auto_encoder(x)
        loss = self.metric(x, x_hat)
        self.auto_encoder.lossLogs = np.append(self.auto_encoder.lossLogs, loss.detach().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch
        x_hat = self.auto_encoder(x)
        loss = self.metric(x, x_hat)
        self.log(f"{prefix}_loss", loss)

    def configure_optimizers(self):
        optimizer = self.auto_encoder.opt
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 3,
                                               verbose= True, min_lr=1.0e-05),
                "monitor": "val_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }



############################# ARCHITECTURES #######################

#AE with Sigmoid activation layer (decoder output layer)
class AutoEncoderSig(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()

        self.lossLogs = np.array([])

    def build_model(self, input_shape, latent_shape):

        self.encoder_layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),

            nn.LeakyReLU(),
            nn.Linear(8, out_features=latent_shape),
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features=latent_shape, out_features=8),
            nn.LeakyReLU(),
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, out_features=input_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encoder(self, x):
        return self.encoder_layers(x)

    def decoder(self, x_):
        return self.decoder_layers(x_)

#AE with SELU as hidden layer activation
class AutoEncoderSelu(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()

        self.lossLogs = np.array([])

    def build_model(self, input_shape, latent_shape):

        self.encoder_layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.Linear(32, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, out_features=latent_shape),
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features=latent_shape, out_features=8),
            nn.SELU(),
            nn.Linear(8, 16),
            nn.SELU(),
            nn.Linear(16, 32),
            nn.SELU(),
            nn.Linear(32, 64),
            nn.SELU(),
            nn.Linear(64, out_features=input_shape),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encoder(self, x):
        return self.encoder_layers(x)

    def decoder(self, x_):
        return self.decoder_layers(x_)


#AE with linear fct as encoder output activation (ReLu in hidden layers)
class AutoEncoderReLU(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()

        self.lossLogs = np.array([])

    def build_model(self, input_shape, latent_shape):

        self.encoder_layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8, out_features=latent_shape)
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features=latent_shape, out_features=8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, out_features=input_shape),
        )


    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encoder(self, x):
        return  self.encoder_layers(x)

    def decoder(self, x_):
        return self.decoder_layers(x_)


#AE with Tanh as decoder output activation
class AutoEncoderTanh(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()

        self.lossLogs = np.array([])

    def build_model(self, input_shape, latent_shape):

        self.encoder_layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, out_features=latent_shape),
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features=latent_shape, out_features=8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, out_features=input_shape),
            nn.Tanh()
        )


    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encoder(self, x):
        return  self.encoder_layers(x)

    def decoder(self, x_):
        return self.decoder_layers(x_)


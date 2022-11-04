import torch
import numpy as np
from torch import nn, optim
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
#Lightning module AE

beta = 0
class Lit_VAE(LightningModule):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae #takes an AE architecture as input
        self.prep = Preprocessing()

    def forward(self, x):
        return self.vae.forward(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, logvar = self.vae(x)

        loss, klLoss, recLoss = self.vae.loss_function(mu, logvar, x, x_hat, beta)

        self.vae.klLoss = np.append(self.vae.klLoss, beta*klLoss.detach().numpy())
        self.vae.recLoss = np.append(self.vae.recLoss, recLoss.detach().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch
        x_hat, mu, logvar = self.vae(x)

        loss, _, _ = self.vae.loss_function(mu, logvar, x, x_hat, beta)
        self.vae.val_loss = np.append(self.vae.val_loss, loss.detach().numpy())

        self.log(f"{prefix}_loss", loss)

    def configure_optimizers(self):
        optimizer = self.vae.opt
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
class VAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.opt = optim.Adam(self.parameters(), lr=1e-3)

        self.klLoss = np.array([])
        self.recLoss = np.array([])
        self.val_loss = np.array([])


        self.ls_shape = None


        self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()

    def build_model(self, input_shape, latent_shape):

        self.ls_shape = latent_shape

        #encoding
        self.encoder_layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=64),
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            #nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            #nn.BatchNorm1d(8),
            nn.LeakyReLU(),
        )
        self.linear1 = nn.Linear(8, out_features=self.ls_shape)
        self.linear2 = nn.Linear(8, out_features=self.ls_shape)


        #decoding
        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features=latent_shape, out_features=8),
            #nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Linear(8, 16),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            #nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, out_features=input_shape),
            nn.Sigmoid()
        )

    def forward(self, x):

        #encoding
        mu, logvar = self.encoder(x)
        z = self.reparametrization(mu, logvar)

        #decoding
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    def encoder(self, x):
        x = self.encoder_layers(x)
        mu =  self.linear1(x)
        logvar = self.linear2(x)

        return mu, logvar

    def decoder(self, x_):
        return self.decoder_layers(x_)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def loss_function(self, mu, logvar, x, x_hat, beta):

        klLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recLoss = ((10*(x - x_hat))**4).sum()
        #recLoss = (((x - x_hat))**2).sum()

        loss = beta*klLoss + recLoss

        return loss, klLoss, recLoss

class VAE_SELU(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.opt = optim.Adam(self.parameters(), lr=1e-3)

        self.klLoss = np.array([])
        self.recLoss = np.array([])
        self.val_loss = np.array([])

        self.ls_shape = None

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()

    def build_model(self, input_shape, latent_shape):
        self.ls_shape = latent_shape

        # encoding
        self.encoder_layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.Linear(32, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
        )
        self.linear1 = nn.Linear(8, out_features=self.ls_shape)
        self.linear2 = nn.Linear(8, out_features=self.ls_shape)

        # decoding
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
            nn.Sigmoid()
        )

    def forward(self, x):
        # encoding
        mu, logvar = self.encoder(x)
        z = self.reparametrization(mu, logvar)

        # decoding
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    def encoder(self, x):
        x = self.encoder_layers(x)
        mu = self.linear1(x)
        logvar = self.linear2(x)

        return mu, logvar

    def decoder(self, x_):
        return self.decoder_layers(x_)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def loss_function(self, mu, logvar, x, x_hat, beta):
        klLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recLoss = ((10*(x - x_hat))**4).sum()
        #recLoss = (((x - x_hat)) ** 2).sum()

        loss = beta * klLoss + recLoss

        return loss, klLoss, recLoss

class VAE_SELU_deep(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.opt = optim.Adam(self.parameters(), lr=1e-3)

        self.klLoss = np.array([])
        self.recLoss = np.array([])
        self.val_loss = np.array([])

        self.ls_shape = None

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()

    def build_model(self, input_shape, latent_shape):
        self.ls_shape = latent_shape

        # encoding
        self.encoder_layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=100),
            nn.SELU(),
            nn.Linear(100, 64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.Linear(32, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
        )
        self.linear1 = nn.Linear(8, out_features=self.ls_shape)
        self.linear2 = nn.Linear(8, out_features=self.ls_shape)

        # decoding
        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features=latent_shape, out_features=8),
            nn.SELU(),

            nn.Linear(8, 16),
            nn.SELU(),
            nn.Linear(16, 32),
            nn.SELU(),
            nn.Linear(32, 64),
            nn.SELU(),
            nn.Linear(64, 100),
            nn.SELU(),
            nn.Linear(100, out_features=input_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encoding
        mu, logvar = self.encoder(x)
        z = self.reparametrization(mu, logvar)

        # decoding
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    def encoder(self, x):
        x = self.encoder_layers(x)
        mu = self.linear1(x)
        logvar = self.linear2(x)

        return mu, logvar

    def decoder(self, x_):
        return self.decoder_layers(x_)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def loss_function(self, mu, logvar, x, x_hat, beta):
        klLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recLoss = ((10*(x - x_hat))**4).sum()
        #recLoss = (((x - x_hat)) ** 2).sum()

        loss = beta * klLoss + recLoss

        return loss, klLoss, recLoss

class VAE_SELU_shallow(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.opt = optim.Adam(self.parameters(), lr=1e-3)

        self.klLoss = np.array([])
        self.recLoss = np.array([])
        self.val_loss = np.array([])

        self.ls_shape = None

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()

    def build_model(self, input_shape, latent_shape):
        self.ls_shape = latent_shape

        # encoding
        self.encoder_layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.SELU(),
        )
        self.linear1 = nn.Linear(32, out_features=self.ls_shape)
        self.linear2 = nn.Linear(32, out_features=self.ls_shape)

        # decoding
        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features=latent_shape, out_features=32),
            nn.SELU(),
            nn.Linear(32, 64),
            nn.SELU(),
            nn.Linear(64, out_features=input_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encoding
        mu, logvar = self.encoder(x)
        z = self.reparametrization(mu, logvar)

        # decoding
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    def encoder(self, x):
        x = self.encoder_layers(x)
        mu = self.linear1(x)
        logvar = self.linear2(x)

        return mu, logvar

    def decoder(self, x_):
        return self.decoder_layers(x_)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def loss_function(self, mu, logvar, x, x_hat, beta):
        klLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recLoss = ((10*(x - x_hat))**4).sum()
        #recLoss = (((x - x_hat)) ** 2).sum()

        loss = beta * klLoss + recLoss

        return loss, klLoss, recLoss

class VAE_shallow(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.opt = optim.Adam(self.parameters(), lr=1e-3)

        self.klLoss = np.array([])
        self.recLoss = np.array([])
        self.val_loss = np.array([])

        self.ls_shape = None

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()

    def build_model(self, input_shape, latent_shape):
        self.ls_shape = latent_shape

        # encoding
        self.encoder_layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )
        self.linear1 = nn.Linear(32, out_features=self.ls_shape)
        self.linear2 = nn.Linear(32, out_features=self.ls_shape)

        # decoding
        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features=latent_shape, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, out_features=input_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encoding
        mu, logvar = self.encoder(x)
        z = self.reparametrization(mu, logvar)

        # decoding
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    def encoder(self, x):
        x = self.encoder_layers(x)
        mu = self.linear1(x)
        logvar = self.linear2(x)

        return mu, logvar

    def decoder(self, x_):
        return self.decoder_layers(x_)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def loss_function(self, mu, logvar, x, x_hat, beta):
        klLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recLoss = ((10*(x - x_hat))**4).sum()
        #recLoss = (((x - x_hat)) ** 2).sum()

        loss = beta * klLoss + recLoss

        return loss, klLoss, recLoss
class Preprocessing():

    def compute_preprocessing_params(self, X):
        self.mu_pp = X.mean(axis=0)
        self.std_pp = X.std(axis=0)
        self.max = X.max()

    def preprocessing(self, X):
        #X_pp = X / self.max * (math.e - 1) + 1  # input between 0 and 1
        #X_pp = np.log(X_pp)

        #X_pp = (X - self.mu_pp) / self.std_pp
        X_pp = X/self.max

        return X_pp






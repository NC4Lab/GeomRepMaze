import torch
import numpy as np
from torch import nn, optim
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

#Lightning module AE
class LitDynAE(LightningModule):
    def __init__(self, network):
        super().__init__()
        self.network = network #takes an AE architecture as input
        self.metric = torch.nn.MSELoss()

        #manual
        #self.automatic_optimization = False

    def forward(self, x):
        return self.network.forward(x)

    def training_step(self, batch, batch_idx):
        x = batch[:, 0, :]
        x_cur = batch[:, 0, :-2]
        x_next = batch[:, 1, :-2]
        x_hat, x_next_hat = self.network(x)

        #loss_AE_net = self.metric(x_cur, x_hat)
        #opt1 = optim.Adam(self.parameters(), lr=1e-3)

        #param for param in model.parameters() if param.requires_grad == True]
        #self.network.opt.zero_grad()
        #self.manual_backward(loss_AE_net)
        #elf.network.opt.step()

        #loss_dyn_net = self.metric(x_next, x_next_hat)
        #self.network.opt.zero_grad()
        #self.manual_backward(loss_dyn_net)
        #self.network.opt.step()


        #self.network.dynLossLogs = np.append(self.network.dynLossLogs, loss_dyn_net.detach().numpy())
        #self.network.AELossLogs = np.append(self.network.AELossLogs, loss_AE_net.detach().numpy())

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch[:, 0, :]

        x_cur = batch[:, 0, :-2]
        x_next = batch[:, 1, :-2]

        x_hat, x_next_hat = self.network(x)
        loss = self.metric(x_next, x_next_hat) + self.metric(x_cur, x_hat)
        self.log(f"{prefix}_loss", loss)

    def configure_optimizers(self):
        optimizer = self.network.opt
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 2,
                                               verbose= True, min_lr=1.0e-04),
                "monitor": "val_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


############################ ARCHITECTURES #######################

#AE with Sigmoid activation layer (decoder output layer)
class dynAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(kwargs["input_shape"], kwargs["latent_shape"])
        self.loss = nn.MSELoss()

        self.dynLossLogs = np.array([])
        self.AELossLogs = np.array([])



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

        self.f_layers = nn.Sequential(
            nn.Linear(latent_shape+2, latent_shape), #add 2 for input goal and step length
            nn.LeakyReLU(),
            nn.Linear(latent_shape, latent_shape),
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
        x_in = x[:, :-2]
        goal = x[:, -2].expand(1, x.shape[0])
        v = x[:, -1].expand(1, x.shape[0])
        latent = self.encoder(x_in)
        rec = self.decoder(latent)


        latent_next = self.f(latent.detach(), goal, v)
        rec_next = self.decoder(latent_next).detach()
        return rec, rec_next

    def encoder(self, x):
        return self.encoder_layers(x)

    def decoder(self, x_):
        return self.decoder_layers(x_)

    def f(self, latent, goal, v):
        return self.f_layers(torch.cat((latent.T, goal, v)).T)

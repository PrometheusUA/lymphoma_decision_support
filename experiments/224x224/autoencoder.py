import torch
from torch import nn
import torch.functional as F
import lightning as L

SUBIMAGE_SIZE = 224

class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        latent_dim: int,
        act_fn: object = nn.ReLU,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, base_channel_size, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(base_channel_size, 2 * base_channel_size, kernel_size=3, stride=2, padding=1), # 224 -> 112
            act_fn(),
            nn.Conv2d(2 * base_channel_size, 4 * base_channel_size, kernel_size=3, stride=2, padding=1), # 112 -> 56
            act_fn(),
            nn.Conv2d(4 * base_channel_size, 8 * base_channel_size, kernel_size=3, stride=2, padding=1), # 56 -> 28
            act_fn(),
            nn.Conv2d(8 * base_channel_size, 16 * base_channel_size, kernel_size=3, stride=2, padding=1), # 28 -> 14
            act_fn(),
            nn.Conv2d(16 * base_channel_size, 32 * base_channel_size, kernel_size=3, stride=2, padding=1), # 14 -> 7
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(32 * base_channel_size * 7 * 7, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        latent_dim: int,
        act_fn: object = nn.ReLU,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7 * base_channel_size),
            nn.Unflatten(1, (32 * base_channel_size, 7, 7)),  # Single feature vector to image grid
            act_fn(),
            nn.ConvTranspose2d(32 * base_channel_size, 16 * base_channel_size, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 => 14x14
            act_fn(),
            nn.ConvTranspose2d(16 * base_channel_size, 8 * base_channel_size, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 => 28x28
            act_fn(),
            nn.ConvTranspose2d(8 * base_channel_size, 4 * base_channel_size, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28 => 56x56
            act_fn(),
            nn.ConvTranspose2d(4 * base_channel_size, 2 * base_channel_size, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56 => 112x112
            act_fn(),
            nn.ConvTranspose2d(2 * base_channel_size, base_channel_size, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x112 => 224x224
            act_fn(),
            nn.Conv2d(base_channel_size, num_input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(L.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = SUBIMAGE_SIZE,
        height: int = SUBIMAGE_SIZE,
        rho: float = 0.1,
        lam1: float = 50.0,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(
            num_input_channels, base_channel_size, latent_dim
        )
        self.decoder = decoder_class(
            num_input_channels, base_channel_size, latent_dim
        )
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
        self.rho = rho
        self.lam1 = lam1

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        # print(z[0].max(), z[0].mean(), z[0].min())
        self.data_rho = z.mean(0)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def _get_kl_reg(self, size_average=True):
        dkl = -self.rho * torch.log(self.data_rho / self.rho) - (
            1 - self.rho
        ) * torch.log(
            (1 - self.data_rho) / (1 - self.rho)
        )  # calculates KL divergence
        rho_loss = 0
        if size_average:
            rho_loss = dkl.mean()
        else:
            rho_loss = dkl.sum()
        return rho_loss

    def _get_l1_reg(self, size_average=True):
        reg_loss = torch.abs(self.data_rho)
        if size_average:
            reg_loss = reg_loss.mean()
        else:
            reg_loss = reg_loss.sum()
        return reg_loss

    def _get_reg_loss(self, size_average=True):
        reg_loss = 0
        reg_loss += self._get_l1_reg(size_average)
        reg_loss *= self.lam1
        return reg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_pure_loss",
        }

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        loss += self._get_reg_loss()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        reg_loss = self._get_reg_loss()
        self.log("val_pure_loss", loss)
        self.log("val_reg_loss", reg_loss)

    def test_step(self, batch, batch_idx):
        # images, classes = batch
        # images = transforms.RandomCrop(SUBIMAGE_SIZE)(images)
        # batch = images, classes
        loss = self._get_reconstruction_loss(batch)
        reg_loss = self._get_reg_loss()
        self.log("test_pure_loss", loss)
        self.log("test_reg_loss", reg_loss)

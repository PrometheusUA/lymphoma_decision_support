import lightning as L
import torch
import torch.functional as F
import torchvision
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn


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
            nn.Conv2d(base_channel_size, base_channel_size, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(
                base_channel_size, base_channel_size, kernel_size=2, stride=2
            ),  # 40x40 => 20x20
            act_fn(),
            nn.Conv2d(
                base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1
            ),
            act_fn(),
            nn.Conv2d(
                2 * base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1
            ),
            act_fn(),
            nn.Conv2d(
                2 * base_channel_size, 2 * base_channel_size, kernel_size=2, stride=2
            ),  # 20x20 => 10x10
            act_fn(),
            nn.Conv2d(
                2 * base_channel_size, 4 * base_channel_size, kernel_size=3, padding=1
            ),
            act_fn(),
            nn.Conv2d(
                4 * base_channel_size, 4 * base_channel_size, kernel_size=3, padding=1
            ),
            act_fn(),
            nn.Conv2d(
                4 * base_channel_size, 4 * base_channel_size, kernel_size=2, stride=2
            ),  # 10x10 => 5x5
            act_fn(),
            nn.Conv2d(4 * base_channel_size, latent_dim, kernel_size=5),
            nn.Flatten(),  # Image grid to single feature vector
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
            nn.Linear(latent_dim, 4 * 5 * 5 * base_channel_size),
            nn.Unflatten(
                1, (4 * base_channel_size, 5, 5)
            ),  # Single feature vector to image grid
            act_fn(),
            nn.Conv2d(
                4 * base_channel_size, 4 * base_channel_size, kernel_size=3, padding=1
            ),
            act_fn(),
            nn.ConvTranspose2d(
                4 * base_channel_size,
                2 * base_channel_size,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 5x5 => 10x10
            act_fn(),
            nn.Conv2d(
                2 * base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1
            ),
            act_fn(),
            nn.ConvTranspose2d(
                2 * base_channel_size,
                base_channel_size,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 10x10 => 20x20
            act_fn(),
            nn.Conv2d(base_channel_size, base_channel_size, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                base_channel_size,
                base_channel_size,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 20x20 => 40x40
            act_fn(),
            nn.Conv2d(base_channel_size, num_input_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 40,
        height: int = 40,
        rho: float = 0.1,
        lam1: float = 50.0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
        self.rho = rho
        self.lam1 = lam1

    def forward(self, x):
        z = self.encoder(x)
        # print(z[0].max(), z[0].mean(), z[0].min())
        self.data_rho = z.mean(0)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def _get_l1_reg(self, size_average=True):
        reg_loss = torch.abs(self.data_rho)
        if size_average:
            reg_loss = reg_loss.mean()
        else:
            reg_loss = reg_loss.sum()
        return reg_loss

    def _get_reg_loss(self, size_average=True):
        reg_loss = self._get_l1_reg(size_average) * self.lam1
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
        loss = self._get_reconstruction_loss(batch)
        reg_loss = self._get_reg_loss()
        self.log("test_pure_loss", loss)
        self.log("test_reg_loss", reg_loss)


class GenerateCallback(Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
            trainer.logger.experiment.add_image(
                "Reconstructions", grid, global_step=trainer.global_step
            )


def get_test_images(dataset, num):
    return torch.stack([dataset[i][0] for i in range(num)], dim=0)


def train(
    train_loader,
    val_loader,
    test_loader,
    dataset_images,
    latent_dim=256,
    lam1=50,
    base_channel_size=32,
    root_dir="./checkpoint/autoencoder",
    logger_additional_name=None,
    checkpoint_path=None,
    max_epochs = 15
):
    logger_version = (
        f"ldim-{latent_dim}_c_hid-{base_channel_size}_lam-{lam1}"
        if not logger_additional_name
        else f"ldim-{latent_dim}_c_hid-{base_channel_size}_lam-{lam1}"
        f"-{logger_additional_name}"
    )
    trainer = L.Trainer(
        default_root_dir=root_dir,
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        log_every_n_steps=20,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(get_test_images(dataset_images, 8), every_n_epochs=1),
            LearningRateMonitor("epoch"),
        ],
        logger=TensorBoardLogger(
            save_dir=root_dir, version=logger_version, name="lightning_logs"
        ),
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    if checkpoint_path:
        model = Autoencoder.load_from_checkpoint(checkpoint_path)
    else:
        model = Autoencoder(
            base_channel_size=base_channel_size, latent_dim=latent_dim, lam1=lam1
        )
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

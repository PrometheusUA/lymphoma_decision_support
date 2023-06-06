import torch
from torch import nn
import torch.functional as F
from tqdm import tqdm
from get_orlov_datasets import get_orlov_datasets
from autoencoder import Autoencoder, Encoder, Decoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import lightning as L
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


PRETRAINED_AUTOENCODER_FILE = './checkpoints/autoencoder/old/ldim-2048_c_hid-32_lam-100-smaller-4/checkpoints/epoch=49-step=7500.ckpt'
SUBIMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_LOADERS_WORKERS = 8
EPOCHS_NUM = 25


class ANet(L.LightningModule):
    def __init__(self):
        super().__init__()
        autoencoder_model = Autoencoder.load_from_checkpoint(PRETRAINED_AUTOENCODER_FILE)
        self.encoder = Encoder(num_input_channels=3, base_channel_size=32, latent_dim=2048)
        self.encoder.load_state_dict(autoencoder_model.encoder.state_dict())
        self.encoder.requires_grad_ = False
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(32, 3)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros(2, 3, SUBIMAGE_SIZE, SUBIMAGE_SIZE)


    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
    
    def _get_loss(self, batch):
        """Given a batch of images, this function returns the loss"""
        x, y = batch  # We do not need the labels
        output = self.forward(x)
        loss = self.criterion(output, y)
        output_labels = torch.argmax(output.cpu(), dim=1)
        acc = accuracy_score(y.cpu(), output_labels)
        prec = precision_score(y.cpu(), output_labels, average='macro', zero_division=0)
        rec = recall_score(y.cpu(), output_labels, average='macro', zero_division=0)
        return loss, acc, prec, rec

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=3, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        loss, acc, prec, rec = self._get_loss(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_precision", prec)
        self.log("train_recall", rec)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, prec, rec = self._get_loss(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", prec)
        self.log("val_recall", rec)

    def test_step(self, batch, batch_idx):
        loss, acc, prec, rec = self._get_loss(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_precision", prec)
        self.log("test_recall", rec)


if __name__ == '__main__':
    train_loader, val_loader, test_loader, additional = get_orlov_datasets(num_loaders_workers=NUM_LOADERS_WORKERS,
                                                                        batch_size=BATCH_SIZE, subimage_size=SUBIMAGE_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ANet()
    model = model.to(device)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float("inf")
    criterion = torch.nn.CrossEntropyLoss()
    lr = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.33, patience=3)
    version = '1.0'

    root_dir = f"./checkpoints/classifier/anet-{version}"
    logger_version = f"anet-{version}-lr-3e-4-dropout-0.8-2"
    trainer = L.Trainer(
        default_root_dir=root_dir,
        accelerator="gpu",
        devices=1,
        max_epochs=EPOCHS_NUM,
        log_every_n_steps=50,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
        ],
        logger=TensorBoardLogger(
            save_dir=root_dir, version=logger_version, name="lightning_logs"
        ),
    )
    trainer.logger._log_graph = (
        True 
    )
    trainer.logger._default_hp_metric = (
        None 
    )
    model = ANet()
    # model = ANet.load_from_checkpoint('C:/_DIPLOMA/code/RESULT_CODE/experiments/224x224/checkpoints/classifier/lightning_logs/ldim-2048_c_hid-32_lam-100-smaller-3/checkpoints/epoch=34-step=5250.ckpt')
    trainer.fit(model, train_loader, val_loader)

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
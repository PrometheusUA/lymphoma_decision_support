from autoencoder import train
from get_orlov_datasets import get_orlov_datasets


if __name__ == '__main__':
    train_loader, val_loader, test_loader, (data, data_test) = get_orlov_datasets()

    train(train_loader, val_loader, test_loader, data.datasets[0], max_epochs=2)

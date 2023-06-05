import os

import numpy as np
import pandas as pd
import torch
import torchvision
from subimage_folder import SubimageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

# orlov dataset can be downloaded here: https://www.kaggle.com/datasets/andrewmvd/malignant-lymphoma-classification
ORLOV_DATASET_FOLDER = "E:/_UNIVER/Diploma/code/RESULT_CODE/data/orlov"
SUBIMAGE_SIZE = 40
BATCH_SIZE = 256
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
TRAIN_SUBIMAGES_NUM = 32
NUM_LOADERS_WORKERS = 10


def get_orlov_datasets(
    orlov_dataset_folder=ORLOV_DATASET_FOLDER,
    train_subimages_num=TRAIN_SUBIMAGES_NUM,
    num_loaders_workers=NUM_LOADERS_WORKERS,
    batch_size=BATCH_SIZE,
):
    torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_val_datas = []
    for i in range(train_subimages_num):
        data_i = SubimageFolder(
            orlov_dataset_folder,
            transform=torchvision.transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(SUBIMAGE_SIZE),
                ]
            ),
        )
        train_val_datas.append(data_i)

    data = torch.utils.data.ConcatDataset(train_val_datas)

    data_test = SubimageFolder(
        orlov_dataset_folder,
        transform=torchvision.transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(SUBIMAGE_SIZE),
            ]
        ),
    )

    assert os.path.isfile(
        f"{orlov_dataset_folder}/patients.csv"
    ), "Patients.csv doesn't exist!"
    patients_df = pd.read_csv(f"{orlov_dataset_folder}/patients.csv", index_col=0)

    train_files, test_files, val_files = [], [], []
    for subset_class in data.datasets[0].classes:
        orlov_patients_class = patients_df[patients_df["class"] == subset_class]

        patients_class_count = len(orlov_patients_class)

        test_split_id = max(1, int(np.floor(TEST_SPLIT * patients_class_count)))
        val_split_id = test_split_id + max(
            1, int(np.floor(VAL_SPLIT * patients_class_count))
        )

        patients_class_test = orlov_patients_class.iloc[:test_split_id]
        patients_class_val = orlov_patients_class.iloc[test_split_id:val_split_id]
        patients_class_train = orlov_patients_class.iloc[val_split_id:]

        for patient_files in patients_class_test.loc[:, "patient_files"]:
            patient_files=eval(patient_files)
            test_files.extend(patient_files)

        for patient_files in patients_class_val.loc[:, "patient_files"]:
            patient_files=eval(patient_files)
            val_files.extend(patient_files)

        for patient_files in patients_class_train.loc[:, "patient_files"]:
            patient_files=eval(patient_files)
            train_files.extend(patient_files)

    train_indices, test_indices, val_indices = [], [], []
    images_array = np.array(data.datasets[0].imgs)[:, 0]
    indices = np.arange(len(images_array))
    for dataset_num in range(len(data.datasets)):
        for filename in train_files:
            filename_index = (
                indices[np.char.find(images_array, filename) >= 0]
                + len(images_array) * dataset_num
            )
            train_indices.extend(filename_index)

        for filename in val_files:
            filename_index = (
                indices[np.char.find(images_array, filename) >= 0]
                + len(images_array) * dataset_num
            )
            val_indices.extend(filename_index)

        for filename in test_files:
            if dataset_num == 0:
                filename_index = (
                    indices[np.char.find(images_array, filename) >= 0]
                    + len(images_array) * dataset_num
                )
                test_indices.extend(filename_index)
            else:
                break

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_loaders_workers,
        persistent_workers=(num_loaders_workers > 0),
    )
    val_loader = DataLoader(
        data,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_loaders_workers,
        persistent_workers=(num_loaders_workers > 0),
    )
    test_loader = DataLoader(
        data_test,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_loaders_workers,
        persistent_workers=(num_loaders_workers > 0),
    )

    return train_loader, val_loader, test_loader, (data, data_test)


if __name__ == "__main__":
    get_orlov_datasets(train_subimages_num=1)

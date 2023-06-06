import numpy as np
import torch
import torchvision
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import crop

SUBIMAGE_SIZE = 224


class SubimageFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        desired_size=SUBIMAGE_SIZE,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.desired_size = desired_size

    def __getitem__(self, index: int):
        image_tensor, class_idx = super().__getitem__(index)
        image_tensor_transformed = image_tensor

        return image_tensor_transformed, class_idx

    def get_image_subimages_batch(self, index: int):
        image_tensor, class_idx = super().__getitem__(index)

        image_batch, reconstruction_info = self.get_pure_image_subimages(image_tensor)

        return image_batch, class_idx, reconstruction_info
    
    def get_pure_image_subimages(self, image_tensor):
        _, image_height, image_width = image_tensor.size()

        assert (
            image_height > self.desired_size and image_width > self.desired_size
        ), "Image should be bigger the its parts"

        count_height = int(np.ceil(image_height / self.desired_size))
        count_width = int(np.ceil(image_width / self.desired_size))

        delta_height = (image_height - self.desired_size) / (count_height - 1)
        delta_width = (image_width - self.desired_size) / (count_width - 1)

        image_batch = []
        for i in range(count_height):
            for j in range(count_width):
                desired_part = crop(
                    image_tensor,
                    int(i * delta_height),
                    int(j * delta_width),
                    self.desired_size,
                    self.desired_size,
                )
                desired_part_transformed = desired_part
                image_batch.append(desired_part_transformed.unsqueeze(0))

        image_batch = torch.cat(image_batch)

        reconstruction_info = (
            delta_height,
            delta_width,
            count_height,
            count_width,
            self.desired_size,
        )

        return image_batch, reconstruction_info

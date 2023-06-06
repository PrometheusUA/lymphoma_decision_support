from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

SUBIMAGE_SIZE = 224


class DatasetTest(Dataset):
    def __init__(
        self, valid_images, wsi, d_height, d_width, subimage_size=SUBIMAGE_SIZE
    ):
        self.valid_images = valid_images
        self.transform = ToTensor()
        self.wsi = wsi
        self.d_height = d_height
        self.d_width = d_width
        self.subimage_size = subimage_size

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        x, y = self.valid_images[idx]

        start_point_wsi = (int(x * self.d_width), int(y * self.d_height))

        wsi_region = self.wsi.read_region(
            start_point_wsi, 0, (self.subimage_size, self.subimage_size)
        ).convert("RGB")

        image = self.transform(wsi_region)

        return (image, x, y)

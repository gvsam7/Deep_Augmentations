import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from typing import Optional


class DataRetrieve(Dataset):
    def __init__(self, ds: Dataset, transforms: Optional[A.Compose] = None):
        super().__init__()
        self.ds = ds
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        image, target = self.ds[idx]
        if self.transforms:
            image = np.array(image)
            image = self.transforms(image=image)["image"]

        return image, target

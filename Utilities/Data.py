import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from typing import Optional
from CutMix.utils import onehot, rand_bbox


class DataRetrieve(Dataset):
    def __init__(self, ds: Dataset, transforms: Optional[A.Compose] = None, augmentations=None, num_class=9, num_mix=1,
                 beta=0.25, prob=0.5):
        super().__init__()
        self.ds = ds
        self.transforms = transforms
        self.augmentations = augmentations
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        image, target = self.ds[idx]

        if self.transforms:
            image = np.array(image)
            image = self.transforms(image=image)["image"]

        if self.augmentations == "cutmix":
            target_onehot = onehot(self.num_class, target)

            for _ in range(self.num_mix):
                r = np.random.rand(1)
                if self.beta <= 0 or r > self.prob:
                    continue

                # generate mixed sample
                lam = np.random.beta(self.beta, self.beta)
                rand_idx = np.random.choice(range(len(self)))

                image2, target2 = self.ds[rand_idx]
                image2 = np.array(image2)
                image2 = self.transforms(image=image2)["image"]
                target2_onehot = onehot(self.num_class, target2)

                bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
                image[bbx1:bbx2, bby1:bby2, :] = image2[bbx1:bbx2, bby1:bby2, :]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
                target_onehot = target_onehot * lam + target2_onehot * (1. - lam)

            return image, target_onehot

        return image, target

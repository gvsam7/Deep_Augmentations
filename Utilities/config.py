import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms(width, height):
    return A.Compose(
        [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            A.Resize(width, height, p=1.),
            A.Cutout(num_holes=8, max_h_size=12, max_w_size=12, fill_value=0, p=0.5),
            ToTensorV2(),
        ]
    )


def val_transforms(width, height):
    return A.Compose(
        [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            A.Resize(width, height),
            ToTensorV2(),
        ]
    )


def test_transforms(width, height):
    return A.Compose(
        [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            A.Resize(width, height),
            ToTensorV2(),
        ]
    )





import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms(width, height, augmentation):
    if augmentation == "position":
        x = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                A.Resize(width, height, p=1.),
                A.ToGray(p=1.),
                A.Rotate(limit=45, p=0.9),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                ToTensorV2(),
                ]
            )
        print("Position Augmentation")
    elif augmentation == "cutout":
        x = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                A.Resize(width, height, p=1.),
                A.Cutout(num_holes=1, max_h_size=12, max_w_size=12, fill_value=0, p=0.5),
                ToTensorV2(),
            ]
        )
        print("Cutout Augmentation")
    else:
        x = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                A.Resize(width, height, p=1.),
                ToTensorV2(),
            ]
        )
        print("No Augmentation")
    return x


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





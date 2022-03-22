import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

__all__ = ["get_transform"]


def get_transform(train, width=1000, height=1000):
    """
    Transformation
    """

    if train:
        if width > 1000:
            width = 1000
        if height > 1000:
            height = 1000

        return A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(limit=(-20, 20), p=0.5),
                A.RandomCrop(width=width, height=height, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.2), p=0.5),
                ToTensorV2(p=1.0),
            ],
            bbox_params={
                "format": "pascal_voc",
                "label_fields": ["labels"],
                "min_area": 100,
            },
        )
    else:
        if width >= 2024 or height >= 2024:
            if width >= height:
                if width > 4000:
                    height = height * 4000 // width
                    width = 4000
                else:
                    height = height * 2024 // width
                    width = 2024
            else:
                if height > 4000:
                    width = width * 4000 // height
                    height = 4000
                else:
                    width = width * 2040 // height
                    height = 2040

        return A.Compose(
            [
                A.Resize(width=width, height=height, p=1.0),
                ToTensorV2(p=1.0),
            ],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )

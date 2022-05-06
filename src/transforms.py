import random

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms as T


class Transforms:
    """Class for getting a Compose of transformations to apply to an image of MTSD."""

    def __init__(self, min_area_train=900, min_area_val=0, img_size_train=1000, img_size_val=None):
        """
        min_area_train: minimum area of bbox to include object for training
        min_area_val: minimum area of bbox to include object for validation
        img_size_train: resulting size of training images
        """
        self.min_area_train = min_area_train
        self.min_area_val = min_area_val
        self.img_size_train = img_size_train
        self.img_size_val = img_size_val

    def get_transform(self, train):
        """
        Get transformations
        train: True if training otherwise testing
        """

        if train:
            return self._transform_train
        else:
            return self._transform_test

    def _transform_train(self, width=1000, height=1000):
        """
        Transformations for train data
        width: resulting width (if img_size_train is not set)
        height: resulting height (if img_size_train is not set)
        """

        if self.img_size_train is not None:
            width = self.img_size_train
            height = self.img_size_train

        return A.Compose(
            [
                A.RandomSizedBBoxSafeCrop(width=width, height=height, p=1.0),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.2), p=0.5),
                ToTensorV2(p=1.0),
            ],
            bbox_params={
                "format": "pascal_voc",
                "label_fields": ["labels"],
                "min_area": self.min_area_train,
                "min_visibility": 0.7,
            },
        )

    def _transform_test(self, width=1000, height=1000):
        do_crop = 1

        if width > 4000 or height > 4000:
            max_size = 4000
        elif width > 2048 or height > 2048:
            max_size = 2048
        else:
            max_size = 1000
            do_crop = 0

        random.seed(123)  # to always get same size

        return A.Compose(
            [
                A.LongestMaxSize(max_size=max_size, p=do_crop),
                ToTensorV2(p=1.0),
            ],
            bbox_params={
                "format": "pascal_voc",
                "label_fields": ["labels"],
                "min_area": self.min_area_val,
            },
        )

    def get_transform_gtsdb(self, train):
        """
        Transformations done to images from GTSDB
        train: True if training otherwise testing
        """

        if train:
            return T.Compose(
                [
                    T.ColorJitter(brightness=0.5, contrast=0.5),
                    T.GaussianBlur(kernel_size=(3, 11), sigma=(0.1, 3)),
                    T.ToTensor(),
                ]
            )
        else:
            return T.Compose([T.ToTensor()])

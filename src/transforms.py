import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms as T
import random


class Transforms:
    def __init__(self, min_area_train=900, min_area_val=0, img_size_train=1000, img_size_val=None):
        self.min_area_train = min_area_train
        self.min_area_val = min_area_val
        self.img_size_train = img_size_train
        self.img_size_val = img_size_val

    def get_transform(self, train):
        if train:
            return self._transform_train
        else:
            return self._transform_test

    def _transform_train(self, width=1000, height=1000):
        if self.img_size_train is not None:
            width, height = self._set_dim(width, height, self.img_size_train)

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
            },
        )

    def _transform_test(self, width=1000, height=1000):
        if self.img_size_val is not None:
            width, height = self._set_dim(width, height, self.img_size_val)

        do_crop = 1 if self.img_size_val is not None else 0

        random.seed(123)

        return A.Compose(
            [A.RandomSizedBBoxSafeCrop(width=width, height=height, p=do_crop), ToTensorV2(p=1.0),],
            bbox_params={
                "format": "pascal_voc",
                "label_fields": ["labels"],
                "min_area": self.min_area_val,
            },
        )

    def _set_dim(self, width, height, target_size):

        if width > target_size or height > target_size:
            if width >= height:
                height = height * target_size // width
                width = target_size
            else:
                width = width * target_size // height
                height = target_size

        return width, height

    def get_transform_gtsdb(self, train):
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


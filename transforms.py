import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


__all__ = ['get_transform']


def get_transform(train):

    if train:
        return A.Compose([
                            A.RandomSizedBBoxSafeCrop(width=640, height=640),
                            A.HorizontalFlip(0.5),
                            # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels'], 'min_area':100})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

                        
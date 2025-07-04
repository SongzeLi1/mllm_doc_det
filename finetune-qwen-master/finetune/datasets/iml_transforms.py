import random

# Augmentation library
import albumentations as albu
import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2
from PIL import Image


class ScaleIfNeeded(DualTransform):
    """只在图像尺寸超出指定大小时进行等比例缩放"""

    def __init__(self, max_height, max_width, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.max_height = max_height
        self.max_width = max_width

    def apply(self, img, **params):
        height, width = img.shape[:2]

        # 只有当图像尺寸超出了指定的输出大小，才进行等比例缩放
        if height > self.max_height or width > self.max_width:
            # 计算缩放比例
            scale = min(self.max_height / height, self.max_width / width)
            new_height, new_width = int(height * scale), int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return img

    def apply_to_mask(self, mask, **params):
        height, width = mask.shape[:2]

        if height > self.max_height or width > self.max_width:
            scale = min(self.max_height / height, self.max_width / width)
            new_height, new_width = int(height * scale), int(width * scale)
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        return mask

    def get_transform_init_args_names(self):
        return ("max_height", "max_width")


def get_albu_transforms(type_="simple", output_size=None, normalize=False):
    """get albumentations transforms

    type_ (str):
        if 'train', then return train transforms with
            random scale, flip, rotate, brightness, contrast, and GaussianBlur augmentation.
        if 'test' then return test transforms
        if 'pad' then return zero-padding transforms
    normalize (bool):
        if True, apply normalization
        if False, skip normalization
    """

    assert type_ in ["augmentation", "simple", "pad", "resize"], "type_ must be 'augmentation' or 'simple' of 'pad' "

    # 定义归一化变换
    normalize_transform = albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else albu.NoOp()

    if type_ == "augmentation":
        trans = albu.Compose(
            [
                albu.RandomScale(scale_limit=0.2, p=1),
                # Flips
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                # Brightness and contrast fluctuation
                albu.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=0.1,
                    p=1,
                ),
                albu.ImageCompression(
                    quality_lower=70,
                    quality_upper=100,
                    p=0.2,
                ),
                # Rotate
                albu.RandomRotate90(p=0.5),
                # Blur
                albu.GaussianBlur(blur_limit=(3, 7), p=0.2),
                # Normalize the image with mean and std (conditional)
                normalize_transform,
            ]
        )
    if type_ == "simple":
        trans = albu.Compose(
            [
                normalize_transform,
            ]
        )
    if type_ == "pad":
        if output_size is None or output_size[0] is None or output_size[1] is None:
            trans = albu.Compose(
                [
                    normalize_transform,
                ]
            )
        else:
            trans = albu.Compose(
                [
                    ScaleIfNeeded(max_height=output_size[0], max_width=output_size[1]),
                    albu.PadIfNeeded(
                        min_height=output_size[0],
                        min_width=output_size[1],
                        border_mode=0,
                        fill=0,
                        position="top_left",
                        fill_mask=0,
                    ),
                    normalize_transform,
                    albu.Crop(0, 0, output_size[0], output_size[1]),
                ]
            )
    if type_ == "resize":
        # 如果 output_size 的任一维度为 None，则不进行 resize 操作
        if output_size is None or output_size[0] is None or output_size[1] is None:
            trans = albu.Compose(
                [
                    normalize_transform,
                ]
            )
        else:
            trans = albu.Compose(
                [
                    albu.Resize(output_size[0], output_size[1]),
                    normalize_transform,
                    albu.Crop(0, 0, output_size[0], output_size[1]),
                ]
            )

    return trans

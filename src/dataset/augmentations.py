# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ALbumentations is a Python library for image augmentation. The Compose function is used to apply                    #
# multiple augmentations at once. To visualize the effects of each algumentation algorithm on an image, please visit: #
# Albumentations Demo: https://github.com/albumentations-team/albumentations-demo                                     #
# Official Albumentations repo: https://github.com/albumentations-team/albumentations                                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from typing import Dict, Tuple

import albumentations as A
import numpy as np


class Augmentations:
    transformer = A.Compose(
        [
            A.Equalize(always_apply=False, p=1.0, mode="cv", by_channels=True),
            A.Flip(always_apply=False, p=0.3),
            A.Blur(always_apply=False, p=0.2, blur_limit=(3, 5)),
            A.CLAHE(always_apply=False, p=0.2, clip_limit=(1, 2), tile_grid_size=(2, 2)),
            A.ChannelShuffle(always_apply=False, p=0.1),
            A.JpegCompression(always_apply=False, p=0.3, quality_lower=50, quality_upper=100),
            A.ElasticTransform(
                always_apply=False,
                p=0.3,
                alpha=1.04,
                sigma=10.07,
                alpha_affine=10.74,
                interpolation=0,
                border_mode=4,
                value=(0, 0, 0),
                mask_value=None,
                approximate=False,
            ),
            A.ShiftScaleRotate(
                always_apply=False,
                p=0.5,
                shift_limit=(0.00, 0.00),
                scale_limit=(0.00, 1.00),
                rotate_limit=(-360, 360),
                interpolation=2,
                border_mode=4,
                value=(0, 0, 0),
                mask_value=None,
            ),
        ]
    )

    @classmethod
    def augment(cls, image: np.ndarray, annotations: Dict, **kwargs: Dict) -> Tuple[np.ndarray, Dict]:
        """Augments an image and its annotations.

        Args:
            image (np.ndarray): The input image.
            annotations (Dict): The annotations associated with the image.
            **kwargs (Dict): Additional keyword arguments. Currently not used. Created for matching with other methods
            called by src.dataset.preprocessing.OrderedCompose class.

        Returns:
            Tuple[np.ndarray, Dict]: A tuple containing the augmented image and the updated annotations.
        """

        return cls.transformer(image=image)["image"], annotations

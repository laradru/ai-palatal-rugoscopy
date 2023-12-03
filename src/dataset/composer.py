# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# A mecanism to apply a list of functions to an image and its annotations. We use this to apply augmentations and     #
# preprocessing functions.                                                                                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from typing import Any, Callable, Dict, List, Tuple

import numpy as np


class OrderedCompose:
    def __init__(self, funcs: List[Callable], **kwargs: Dict) -> None:
        """Class constructor. Compose a list of functions and apply them to an image and its annotations via __call__.

        Args:
            funcs (List[Callable]): A list of functions to apply to an image and its annotations.
            **kwargs (Dict): Optional keyword arguments. Useful for functions that require additional parameters.
        """

        self.funcs = funcs
        self.kwargs = kwargs

    def __call__(self, image: np.ndarray, annotations: Dict) -> Any:
        """Apply the functions in the list to an image and its annotations.

        Args:
            image (np.ndarray): The input image.
            annotations (Dict): The image annotations.
        Returns:
            Any: A preprocessed image and its corresponding annotations.
        """

        for func in self.funcs:
            image, annotations = func(image, annotations, **self.kwargs)

        return image, annotations


def join_patches(patches: List[np.ndarray], filenames: List[str]) -> np.ndarray:
    if len(patches) == 0:
        raise ValueError("Input list of patches is empty.")
    
    dimensions = [x_y_from_filename(filename) for filename in filenames]
    result_shape = calculate_result_shape_from_patches(patches, dimensions)
    result_image = np.zeros(result_shape, dtype=np.uint8)

    for (x_result_image, y_result_image), patch in zip(dimensions, patches):
        height, width = patch.shape

        result_image[
            x_result_image:x_result_image+height, 
            y_result_image:y_result_image+width
        ] = np.maximum(
            result_image[x_result_image:x_result_image+height, y_result_image:y_result_image+width],
            patch
            )

    return result_image


def calculate_result_shape_from_patches(patches: List[np.ndarray], dimensions: Tuple[int, int]) -> Tuple[int, int]:
    max_x = 0
    max_y = 0
    for (x, y), patch in zip(dimensions, patches):
        patch_height, patch_width = patch.shape
        if x + patch_width > max_x: max_x = x + patch_width
        if y + patch_height > max_y: max_y = y + patch_height

    return (max_x, max_y)


def x_y_from_filename(filename: str) -> Tuple[int, int]:
    split_by_underscore = filename.split("_")
    x = int(split_by_underscore[1])
    y = int(split_by_underscore[2].split(".")[0])
    
    return (x, y)
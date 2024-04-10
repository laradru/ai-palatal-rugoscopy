# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# A mecanism to apply a list of functions to an image and its annotations. We use this to apply augmentations and     #
# preprocessing functions.                                                                                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from typing import Any, Callable, Dict, List

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

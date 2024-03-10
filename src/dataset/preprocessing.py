# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Basics for preprocessing images and annotations.                                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from ctypes import ArgumentError
from typing import Dict, Tuple

import cv2
import numpy as np


class CocoPreprocessing:
    @staticmethod
    def crop(image: np.ndarray, annotations: Dict, **kwargs: Dict) -> Tuple[np.ndarray, Dict]:
        """Crop an image based on the given annotations.

        Args:
            image (np.ndarray): The input image.
            annotations (Dict): The annotations containing the bounding box coordinates.
            **kwargs (Dict): Additional keyword arguments.
                format (str): The format of the image. If "channel_last", the image is assumed to be in the format
                    (height, width, channels). Otherwise, the image is assumed to be in the format
                    (channels, height, width).
        Returns:
            Tuple[np.ndarray, Dict]: The cropped image and the updated annotations.
        """

        bbox = np.array(annotations["bbox"], dtype=np.int32)

        if "format" in kwargs.keys():
            format = kwargs["format"]

            if format == "channel_last":
                return image[bbox[1] : (bbox[1] + bbox[3]), bbox[0] : (bbox[0] + bbox[2]), :], annotations

        return image[:, bbox[1] : (bbox[1] + bbox[3]), bbox[0] : (bbox[0] + bbox[2])], annotations

    @staticmethod
    def resize(image: np.ndarray, annotations: Dict, **kwargs: Dict) -> Tuple[np.ndarray, Dict]:
        """Resizes an image using the OpenCV `cv2.resize` function.

        Args:
            image (np.ndarray): The image to be resized.
            annotations (Dict): The annotations associated with the image.
            **kwargs (Dict): Additional keyword arguments for the `cv2.resize` function.
                size (Tuple[int, int]): The size to which the image should be resized.

        Returns:
            Tuple[np.ndarray, Dict]: A tuple containing the resized image and the annotations.
        Raises:
            ArgumentError: If the size is not specified.
        """

        if "size" in kwargs.keys():
            size = kwargs["size"]
            return cv2.resize(image, size), annotations
        else:
            raise ArgumentError("Size is not specified.")

    @staticmethod
    def resize_with_factor(image: np.ndarray, annotations: Dict, **kwargs: Dict) -> Tuple[np.ndarray, Dict]:
        """Resize the input image and annotations by a given factor.

        Args:
            image (np.ndarray): The input image to be resized.
            annotations (Dict): The annotations associated with the image.
            **kwargs (Dict): Additional keyword arguments.
                resize_factor (float): The factor by which the image and annotations should be resized.

        Returns:
            Tuple[np.ndarray, Dict]: The resized image and updated annotations.
        """

        factor = kwargs["resize_factor"]
        return cv2.resize(image, None, fx=factor, fy=factor), annotations

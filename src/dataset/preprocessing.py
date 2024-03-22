# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Basics for preprocessing images and annotations.                                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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
    def resize_to_target(image: np.ndarray, annotations: Dict, **kwargs: Dict) -> Tuple[np.ndarray, Dict]:
        """Resize the given image and its annotations to the target size.

        Args:
            image (np.ndarray): The input image to be resized.
            annotations (Dict): The annotations associated with the image.
            **kwargs (Dict): Additional keyword arguments.

        Returns:
            Tuple[np.ndarray, Dict]: A tuple containing the resized image and updated annotations.
        """

        target = kwargs["resize_target"]

        rows, cols = image.shape[:2]
        resize_factor = target / max(rows, cols)

        # Update annotations
        if annotations:
            for instance in range(len(annotations)):
                bbox, seg = np.array(annotations[instance]["bbox"]), np.array(annotations[instance]["segmentation"])
                annotations[instance]["bbox"] = (bbox * resize_factor).tolist()
                annotations[instance]["segmentation"] = (seg * resize_factor).tolist()

        return cv2.resize(image, None, fx=resize_factor, fy=resize_factor), annotations

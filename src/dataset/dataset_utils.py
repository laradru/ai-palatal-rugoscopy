# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Common dataset utility functions                                                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from typing import List, Tuple

import cv2
import numpy as np


@staticmethod
def read_image(image_path: str, read_mode: int = cv2.IMREAD_COLOR, channel_first: bool = False) -> np.ndarray:
    """Reads an image from the given path.

    Args:
        image_path (str): The path of the image file.
        read_mode (int, optional): The mode used to read the image. Defaults to cv2.IMREAD_COLOR (BGR image).
        channel_first (bool, optional): Whether to return the image with channel first format (C, H, W).
        Otherwise return the image with channel last format (H, W, C). Defaults to False.
    Returns:
        np.ndarray: The image read from the file.
    Raises:
        ValueError: If the image path does not exist.
    """

    if not os.path.exists(image_path):
        raise ValueError(f"Path {image_path} does not exist.")

    image = cv2.imread(image_path, read_mode)

    if channel_first:
        return image.transpose(2, 0, 1)
    else:
        return image


@staticmethod
def read_paths(directory_path: str) -> List[str]:
    """Read the list of paths in a given directory.

    Args:
        directory_path (str): The path of the directory to read.
    Returns:
        List[str]: A list of paths in the directory.
    Raises:
        ValueError: If the given path is not a directory.
    """

    if not os.path.isdir(directory_path):
        raise ValueError(f"Path {directory_path} is not a directory.")

    return os.listdir(directory_path)


@staticmethod
def join_patches(patches: List[np.ndarray], filenames: List[str]) -> np.ndarray:
    """Joins a list of patches into a single image.

    Args:
        patches (List[np.ndarray]): The patches to join.
        filenames (List[str]): The filenames for the patches in the format <name>_x_y.<ext> 
        where x and y represent a position of the patch in the original image.

    Returns:
        np.ndarray: A single image with all the patches merged together. When pixels collide
        in two or more patches, the greather value is used.

    Raises:
        ValueError: If no patches are provided as input.
    """
    if len(patches) == 0:
        raise ValueError("Input list of patches is empty.")
    
    patches = [patch.astype(np.uint8) for patch in patches]
    patches_positions = [x_y_from_filename(filename_from_path(filename)) for filename in filenames]
    result_shape = calculate_result_shape_from_patches(patches, patches_positions)
    result_image = np.zeros(result_shape, dtype=np.uint8)

    for (init_y, init_x), patch in zip(patches_positions, patches):
        row, col = patch.shape
        result_image[init_y:init_y + row, init_x:init_x + col] |= patch

    return result_image


def calculate_result_shape_from_patches(patches: List[np.ndarray], dimensions: Tuple[int, int]) -> Tuple[int, int]:
    """Given a set of patches, calculates the resulting size of an image that contains all patches.

    Args:
        patches (List[np.ndarray]): The patches involved in the operation.
        dimensions (Tuple[int, int]): A list of coordinates (x, y) of where, in the original images,
        each patch is positioned.

    Returns:
        Tuple[int, int]: The size of each dimension of the resulting calculated image.
    """
    coords = list(zip(*dimensions))
    shapes = list(zip(*[patch.shape[:2] for patch in patches]))
    image_shape = np.max(np.array(shapes) + np.array(coords), axis=1)

    return image_shape


def x_y_from_filename(filename: str) -> Tuple[int, int]:
    """Extracts x and y coordinates from a filename in the format <name>_x_y.<ext>.

    Args:
        filename (str): The name of the file in the expected format.

    Returns:
        Tuple[int, int]: The x and y coordinates extracted from the filename.
    """
    split_by_underscore = filename.split("_")
    x = int(split_by_underscore[1])
    y = int(split_by_underscore[2].split(".")[0])

    return (x, y)

def filename_from_path(path: str) -> str:
    """Extracts filename from a path. Robus enough to receive a filename and
    return if if not a path.
    
    Args:
        path (str): A path to a file.
        
    Returns:
        str: The extracted filename.
    """
    basename = os.path.basename(path)
    filename, _ = os.path.splitext(basename)
    return filename
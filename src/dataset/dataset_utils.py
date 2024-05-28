# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Common dataset utility functions                                                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from typing import Dict, Generator, List, Tuple

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from src.dataset.annotations_utils import smooth_annotations


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


def generate_binary_component(image: np.ndarray, annotation: Dict) -> np.ndarray:
    """Generate an instance mask for the given annotation.

    Args:
        image (np.ndarray): The image array.
        annotation (Dict): A dictionary containing a single annotation.

    Returns:
        np.ndarray: The instance mask.
    """

    # There are some annotations that correspond to a crowd of objects (e.g. crowd of people).
    # In such cases, the segmentation is not a polygon, but a Run-length Encoding (RLE). A RLE is basicaly composed by
    # a list of values followed by the number of occurences that this value appears sequentially.
    # Pycocotools package provides functions to encode and decode RLEs.

    if annotation["iscrowd"] == 1:
        mask = coco_mask.decode(annotation["segmentation"])
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygons = np.array(annotation["segmentation"], dtype=object)

        # Sometimes an object can be composed by multiple polygons.
        if polygons.shape[0] > 1:
            for polygon in polygons:
                polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
                mask = cv2.fillPoly(mask, [polygon], color=1)
        else:
            polygon = polygons.reshape((-1, 2)).astype(np.int32)
            mask = cv2.fillPoly(mask, [polygon], color=1)

    return mask


def generate_binary_mask(image: np.ndarray, annotations: Dict) -> np.ndarray:
    """Generates a binary mask based on the given image and annotations.

    Args:
        image (np.ndarray): The input image.
        annotations (Dict): The dictionary containing the annotations.

    Returns:
        np.ndarray: The binary mask generated from the image and annotations.
    """

    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for instance_annotation in annotations:
        mask = generate_binary_component(image, instance_annotation)
        binary_mask = np.bitwise_or(binary_mask, mask)

    return binary_mask


def generate_category_mask(image: np.ndarray, annotations: Dict) -> np.ndarray:
    """Generate a category mask based on the given image and annotations. A category mask is a mask that contains 0 as
    background and the category id as foreground.

    Args:
        image (np.ndarray): The input image.
        annotations (Dict): The annotations containing instance information.

    Returns:
        np.ndarray: The generated category mask.

    """
    category_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for instance_annotation in annotations:
        mask = generate_binary_component(image, instance_annotation)
        category_mask += mask * instance_annotation["category_id"]

    return category_mask


def extract_bbox_segmentation(instance_mask: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Extracts the bounding box and segmentation of an instance mask.

    Args:
        instance_mask (np.ndarray): The instance mask to extract the bounding box and segmentation from.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing the instance bounding box and segmentation.
            The instance bounding box is a list of integers representing [x, y, width, height].
            The instance segmentation is a list of integers representing [x1, y1, x2, y2, x3, y3, ...].
    """

    instance_contours, _ = cv2.findContours(instance_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    instance_bbox = list(cv2.boundingRect(instance_contours[0]))

    # Now, organize instance segmentation into a list of [x1, y1, x2, y2, x3, y3, ...]
    instance_contours = instance_contours[0].reshape(-1, 2).astype(int)
    instance_segmentation = [0] * (instance_contours.shape[0] * instance_contours.shape[1])
    instance_segmentation[::2] = instance_contours[:, 0]
    instance_segmentation[1::2] = instance_contours[:, 1]

    return instance_bbox, instance_segmentation


def patch_generator(image: np.ndarray, patch_size: int, stride: int) -> Generator:
    """Generate patches from an image using a sliding window approach.
    Patch size is fixed. Patches that would be smaller are filled with zeros.

    Args:
        image (np.ndarray): The input image.
        patch_size (int): The size of the patches.
        stride (int): The stride between patches.

    Yields:
        Generator[np.ndarray, Tuple[int, int]]: A generator that yields a patch and its coordinates.

    Raises:
        StopIteration: Raised when generator is closed.
    """

    rows, cols = image.shape[:2]
    patch_shape = (patch_size, patch_size, image.shape[2]) if image.ndim > 2 else (patch_size, patch_size)
    cur_row = 0
    cur_col = 0

    while cur_row < rows:
        coord = (cur_row, cur_col)

        patch = np.zeros(patch_shape, dtype=image.dtype)
        patch_rows = min(patch_size, rows - cur_row)
        patch_cols = min(patch_size, cols - cur_col)
        patch[:patch_rows, :patch_cols] = image[cur_row : cur_row + patch_rows, cur_col : cur_col + patch_cols]

        if cur_col + patch_size >= cols - 1:
            previous_was_inside = cur_row - stride + patch_size < rows - 1

            cur_row += stride if previous_was_inside else rows  # Force exit if the last patch hasn't patch_size rows.
            cur_col = 0
        else:
            cur_col += stride

        yield patch, coord
    raise StopIteration()


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
        result_image[init_y : init_y + row, init_x : init_x + col] |= patch

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


def custom_collate(data):
    """Collates a list of data instances into separate lists of images and annotations.

    Args:
        data (list): A list of data instances, where each instance is a tuple containing an image and its annotation.

    Returns:
        tuple: A tuple containing two lists - the list of images and the list of annotations.
    """

    imgs = []
    annotations = []

    for instance in data:
        imgs.append(instance[0])
        annotations.append(instance[1])

    return imgs, annotations


def to_cvat(predictions: Dict, categories: Dict[str, str]) -> List[Dict]:
    """Converts predictions from a model to the CVAT format.

    Args:
        predictions (Dict): A dictionary containing the model predictions. It should have the following keys:
            - "masks": A numpy array representing the masks of the predictions.
            - "labels": A numpy array representing the labels of the predictions.
            - "scores": A numpy array representing the scores of the predictions.
        categories (Dict[str, str]): A dictionary mapping labels to their corresponding names.

    Returns:
        List[Dict]: A list of dictionaries representing the predictions in the CVAT format.
        Each dictionary contains the following keys:
            - "confidence": The confidence score of the prediction.
            - "label": The name of the predicted label.
            - "points": The segmentation points of the prediction.
            - "type": The type of the prediction (always "polygon" in this case).
    """

    res = []

    for mask, label, score in zip(predictions["masks"], predictions["labels"], predictions["scores"]):
        if np.any(mask > 0):
            label = str(label)
            __, segmentation = extract_bbox_segmentation(mask)
            segmentation = list(map(float, segmentation))
            segmentation = smooth_annotations(segmentation)
            res.append({"confidence": score, "label": categories[label], "points": segmentation, "type": "polygon"})

    return res

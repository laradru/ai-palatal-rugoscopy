# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Extra functions useful for handling annotations.                                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def to_dataframe(dictionary: Dict) -> pd.DataFrame:
    """Convert a dictionary into a Pandas DataFrame.

    Args:
        dictionary (Dict): The dictionary to be converted.
    Returns:
        pd.DataFrame: The resulting DataFrame.
    """

    return pd.DataFrame(data=dictionary)


def to_dict(data: List[Dict], key_type: str) -> Dict:
    """Convert a list of dictionaries into a dictionary of lists using a specified key type.

    Args:
        data (List[Dict]): The list of dictionaries to be converted.
        key_type (str): The key type to be used for grouping the dictionaries.
    Returns:
        Dict: A dictionary of lists, where the key is the specified key type and the value is a list
        of dictionaries with the same key type.
    Raises:
        ValueError: If the specified key type is not present in the dictionaries.
    Example:
        data = [
            {"name": "John", "age": 25},
            {"name": "Jane", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        key_type = "age"
        result = to_dict(data, key_type)
        # Result: {25: [{"name": "John", "age": 25}, {"name": "Bob", "age": 25}], 30: [{"name": "Jane", "age": 30}]}
    """

    if key_type not in data[0].keys():
        raise ValueError("Invalid key")

    data_dictionary = {}
    data = sorted(data.copy(), key=lambda x: x[key_type])

    for key, group in itertools.groupby(data, lambda x: x[key_type]):
        data_dictionary[key] = list(group)

    return data_dictionary


def xywh_to_xyxy(bbox: List) -> List:
    """Converts a bounding box in the format [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        bbox (List): The bounding box in the format [x, y, w, h].
    Returns:
        List: The bounding box in the format [x1, y1, x2, y2].
    """

    x, y, w, h = bbox

    x2 = x + w
    y2 = y + h

    return [x, y, x2, y2]


def really_agnostic_segmentation_nms(masks: List[np.ndarray], scores: List[float], threshold: float) -> List[int]:
    """Perform a custom version of non-maximum suppression algorithm, where segmentation is the main criteria.
    Also, instead of computing the Intersection over Union (IoU), we compare the area of the masks.

    Args:
        masks (List[np.ndarray]): A list of numpy arrays representing the masks.
        scores (List[float]): A list of floats representing the scores of the masks.
        threshold (float): A float value representing the threshold for intersection over union (IoU) between masks.

    Returns:
        List[int]: A list of integers representing the indices of the remaining masks after suppression.
    """

    sorted_args = np.argsort(scores)[::-1]
    to_remove = set()

    for i in range(len(sorted_args) - 1):
        if i in to_remove:
            continue

        mask_x = masks[sorted_args[i]]
        mask_x_area = mask_x.sum()

        intersections = [np.logical_and(mask_x, masks[j]).sum() for j in sorted_args[i + 1 :]]
        min_area = [min(mask_x_area, masks[j].sum()) for j in sorted_args[i + 1 :]]
        ious = np.divide(intersections, min_area)

        to_remove.update(sorted_args[np.where(ious >= threshold)[0] + (i + 1)])
    return list(set(sorted_args) - to_remove)


def smooth_annotations(segmentation: List[int], gaussian_sigma: float = 3) -> List[int]:
    """Smooth the given segmentation by applying a Gaussian filter to the x and y coordinates.

    Args:
        segmentation (List[int]): The segmentation to be smoothed. It should be a list of integers representing
        the x and y coordinates.
        gaussian_sigma (float, optional): The standard deviation of the Gaussian filter. Defaults to 3.

    Returns:
        List[int]: The smoothed segmentation. It is a new list of integers representing the x and y coordinates.
    """

    xs, ys = segmentation[0::2], segmentation[1::2]

    # smooth coordinates
    xs = gaussian_filter1d(xs, sigma=gaussian_sigma)
    ys = gaussian_filter1d(ys, sigma=gaussian_sigma)

    # roll coordinates so extremities can be smoothed as well.
    xs = gaussian_filter1d(np.roll(xs, 5), sigma=gaussian_sigma)
    ys = gaussian_filter1d(np.roll(ys, 5), sigma=gaussian_sigma)

    # create new segmentation
    new_annotation = [0] * len(xs) * 2
    new_annotation[0::2] = xs
    new_annotation[1::2] = ys

    return new_annotation

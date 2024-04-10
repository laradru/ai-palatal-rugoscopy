# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# A COCO dataset class that should be used with PyTorch.                                                              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from scipy.stats import mode
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.annotations_coco import COCOAnnotations
from src.dataset.annotations_utils import to_dict, xywh_to_xyxy
from src.dataset.dataset_base import MutableDataset
from src.dataset.dataset_utils import (
    custom_collate,
    extract_bbox_segmentation,
    generate_binary_component,
    generate_binary_mask,
    generate_category_mask,
    patch_generator,
    read_image,
)
from src.dataset.preprocessing import CocoPreprocessing


@dataclass
class CocoDataset(MutableDataset):
    data_directory_path: str
    data_annotation_path: str
    augmentations: Callable = None
    preprocessing: Callable = None
    seed: Any | None = None
    balancing_strategy: str | None = None

    def __post_init__(self) -> None:
        """Initialize important attributes of the object. This function is automatically called after the
        object has been created."""

        if self.seed is not None:
            np.random.seed(self.seed)

        super().__init__()

        self.tree = COCOAnnotations(self.data_annotation_path)
        self.preview_dataset()

    def dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        """Class method that returns a DataLoader object.

        Args:
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data or not.

        Returns:
            DataLoader: The DataLoader object.
        """

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)

    def preview_dataset(self) -> None:
        """Prints a preview of the dataset by displaying some dataset statistics."""

        horizontal_bar_length = 120
        categories_str = [f"{c['id']}: {c['name']}" for c in self.tree.data["categories"]]

        print("=" * horizontal_bar_length)
        print(f"Dataset categories: {categories_str}")
        print(f"Number of images: {len(self.tree.data['images'])}")
        print(f"Number of Annotations: {len(self.tree.data['annotations'])}")
        print("=" * horizontal_bar_length)
        print("Per-category info:")

        images_per_category = to_dict(self.tree.data["annotations"], "category_id")

        for c in self.tree.data["categories"]:
            try:
                print(f"Category Label: {c['name']} \t Category ID: {c['id']}")
                print(f"Instances: {len(images_per_category[c['id']])}")
            except KeyError:
                print(f"Instances: {0}")
        print("=" * horizontal_bar_length)


class CocoDatasetClassification(CocoDataset):
    def __post_init__(self) -> None:
        """Initialize important attributes of the object. This function is automatically called after the
        object has been created."""

        super().__post_init__()

        self.images = to_dict(self.tree.data["images"], "id")
        self.categories = to_dict(self.tree.data["categories"], "id")
        self.annotations = self.tree.data.get("annotations")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image and category data for a given index.

        Args:
            idx (int): The index of the data to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor and the category tensor.
        """

        annotation = self.annotations[idx]
        image_data = deepcopy(self.images[annotation["image_id"]])
        image_path = os.path.join(self.data_directory_path, image_data[0]["file_name"])
        image = read_image(image_path)

        # apply preprocessing
        if self.preprocessing is not None:
            image, annotation = self.preprocessing(image, annotation)

        # apply augmentations
        if self.augmentations is not None:
            image, annotation = self.augmentations(image, annotation)

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (H, W, C) -> (C, H, W)
        category = torch.tensor(self.categories[annotation["category_id"]][0]["id"] - 1, dtype=torch.float32)

        return image, category

    def __len__(self) -> int:
        """Returns the length of the object.

        Returns:
            int: The length of the object.
        """

        return len(self.annotations)


class CocoDatasetInstanceSegmentation(CocoDataset):
    def __post_init__(self) -> None:
        """Initialize important attributes of the object. This function is automatically called after the
        object has been created."""

        super().__post_init__()

        self.images = self.tree.data.get("images")
        self.categories = to_dict(self.tree.data["categories"], "id")
        self.annotations = to_dict(self.tree.data.get("annotations"), "image_id")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Retrieve an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, Dict]: A tuple containing the image tensor and a dictionary of targets.
                - The image tensor represents the preprocessed input image.
                - The dictionary of targets contains the following keys:
                    - "boxes": A list of bounding box coordinates.
                    - "labels": A list of category labels.
                    - "masks": A list of instance masks.
        """

        image_data = self.images[idx]
        annotations = deepcopy(self.annotations[image_data["id"]])
        image_path = os.path.join(self.data_directory_path, image_data["file_name"])
        image = read_image(image_path)

        # apply preprocessing
        if self.preprocessing is not None:
            image, annotations = self.preprocessing(image, annotations)

        # apply augmentations
        if self.augmentations is not None:
            image, annotations = self.augmentations(image, annotations)

        # Generate instance masks for each annotation
        targets = {"boxes": [], "labels": [], "masks": []}

        for annotation in annotations:
            mask = generate_binary_component(image, annotation)
            targets["masks"].append(mask)
            targets["boxes"].append(xywh_to_xyxy(annotation["bbox"]))
            targets["labels"].append(annotation["category_id"] - 1)

        targets["masks"] = torch.tensor(np.array(targets["masks"]), dtype=torch.uint8)
        targets["boxes"] = torch.tensor(np.array(targets["boxes"]), dtype=torch.float64)
        targets["labels"] = torch.tensor(np.array(targets["labels"]), dtype=torch.int64)

        return T.ToTensor()(image), targets

    def __len__(self) -> int:
        """Returns the length of the object.

        Returns:
            int: The length of the object.
        """

        return len(self.images)

    def extract_patches(self, output_dir: str, patch_size: int, stride: int, min_area_percent: float, **kwargs) -> None:
        """Extracts patches from images and saves them along with their annotations.

        Args:
            output_dir (str): The directory where the patches and annotations will be saved.
            patch_size (int): The size of the patches.
            stride (int): The stride between patches.
            min_area_percent (float): The minimum area percentage (<= 1) for a patch to be considered valid.
        """

        # Initialize patch annotations. Categories are the same as the image annotations.
        patch_annotations = COCOAnnotations.from_dict(
            {
                "categories": self.tree.data["categories"],
                "images": [],
                "annotations": [],
            }
        )

        image_id = 1
        annotation_id = 1
        images_output_dir = os.path.join(output_dir, "images")
        annotations_output_dir = os.path.join(output_dir, "annotations")
        min_area = patch_size * patch_size * min_area_percent
        resize_image_width = kwargs.get("resize_image_width", None)
        rfactor = 1.0

        # Create output subdirectories.
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(annotations_output_dir, exist_ok=True)

        for image_data in tqdm(self.images):
            image_path = os.path.join(self.data_directory_path, image_data["file_name"])

            if image_data["id"] not in self.annotations.keys():
                continue

            image_annotations = self.annotations[image_data["id"]]
            image = read_image(image_path)

            # ~~ Extend mask dimensions to match with patch_size and stride values.
            extended_image = np.zeros(extended_dimensions(patch_size, stride, image.shape[:2]))

            # ~~ Draw components on masks based on the image annotations made on CVAT.
            binary_mask = generate_binary_mask(extended_image, image_annotations)
            category_mask = generate_category_mask(extended_image, image_annotations)
            __, component_mask = cv2.connectedComponents(binary_mask)
            component_mask = component_mask.astype(binary_mask.dtype)

            if resize_image_width is not None:
                rfactor = resize_image_width / max(image.shape[0], image.shape[1])

                image, __ = CocoPreprocessing.resize_with_factor(image, None, resize_factor=rfactor)
                binary_mask, __ = CocoPreprocessing.resize_with_factor(binary_mask, None, resize_factor=rfactor)
                category_mask, __ = CocoPreprocessing.resize_with_factor(category_mask, None, resize_factor=rfactor)
                component_mask, __ = CocoPreprocessing.resize_with_factor(component_mask, None, resize_factor=rfactor)

            # ~~ Extract patches
            patch_gen = patch_generator(binary_mask, patch_size, stride)

            try:
                while patch_data := next(patch_gen):
                    patch, coord = patch_data
                    patch_basename, ext = os.path.splitext(os.path.basename(image_data["file_name"]))
                    patch_name = f"{patch_basename}_{str(coord[0])}_{str(coord[1])}{ext}"
                    patch_rows, patch_cols = patch.shape[0], patch.shape[1]

                    # Crop map to get components inside the patch
                    patch_map = component_mask[coord[0] : coord[0] + patch_rows, coord[1] : coord[1] + patch_cols]

                    # Ignore patches without any components...
                    if np.all(patch_map == 0):
                        continue

                    # ... or with less than min_area
                    if any([area < min_area * rfactor for area in np.bincount(patch_map.flatten())[1:]]):
                        continue

                    patch_annotations.add_image_instance(image_id, patch_name, patch_rows, patch_cols)

                    patch_category_mask = category_mask[
                        coord[0] : coord[0] + patch_rows,
                        coord[1] : coord[1] + patch_cols,
                    ]

                    patch_image = image[
                        coord[0] : coord[0] + patch_rows,
                        coord[1] : coord[1] + patch_cols,
                    ]

                    # Extract data for each component and create a new annotation instance.
                    for label in np.unique(patch_map)[1:]:  # 0 is background
                        instance_map = np.array(patch_map == label, dtype=np.uint8)
                        instance_category = mode(patch_category_mask[patch_map == label], axis=None, keepdims=False)[0]
                        instance_bbox, instance_segmentation = extract_bbox_segmentation(instance_map)

                        patch_annotations.add_annotation_instance(
                            id=annotation_id,
                            image_id=image_id,
                            category_id=int(instance_category),
                            bbox=instance_bbox,
                            segmentation=[list(np.array(instance_segmentation, dtype=float))],
                            iscrowd=0,
                        )

                        annotation_id += 1

                    cv2.imwrite(os.path.join(output_dir, "images", patch_name), patch_image)
                    image_id += 1

            except (RuntimeError, StopIteration):
                pass

        patch_annotations.save(output_path=os.path.join(output_dir, "annotations", "annotations.json"))

    def split(self, *percentages: float, random: bool) -> Tuple[Any, ...]:
        """Splits the dataset into subsets based on the given percentages.

        Args:
            *percentages (float): The percentages to split the dataset into subsets. The sum of percentages must be
            equal to 1.
            random (bool): Determines whether to shuffle the images before splitting.
        Returns:
            Tuple: A tuple of subsets, each containing a portion of the dataset.
        """

        assert np.sum([*percentages]) == 1, "Summation of percentages must be equal to 1."

        subsets = []
        all_images_ids = [
            img["id"]
            for img in self.images
            if self.annotations.get(img["id"]) is not None and len(self.annotations[img["id"]]) > 0
        ]

        total_images = len(all_images_ids)
        subset_sizes = [int(total_images * perc) for perc in percentages]

        if random:
            np.random.shuffle(all_images_ids)

        for ss in subset_sizes:
            if ss == 0:  # Skip empty subsets
                continue

            subset = deepcopy(self)
            indexes = all_images_ids[:ss]

            subset.tree.data["images"] = [
                self.images[j] for i in indexes for j in range(len(self.images)) if self.images[j]["id"] == i
            ]
            subset.images = to_dict(subset.tree.data["images"], "id")

            image_annotations = to_dict(subset.tree.data["annotations"], "image_id")
            subset.tree.data["annotations"] = [image_annotations[image_id] for image_id in subset.images.keys()]
            subset.tree.data["annotations"] = [item for sublist in subset.tree.data["annotations"] for item in sublist]
            subset.annotations = subset.tree.data.get("annotations")

            subsets.append(subset)
            all_images_ids = all_images_ids[ss:]

        return tuple(subsets)


def extended_dimensions(patch_size: int, stride: int, image_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate the extended dimensions of an image based on the patch size and stride.

    If for a given dimension, considering the patch_size and the stride, the image will perfectly fit
    all patches, nothing is done.

    However, if a dimension doesn't fit patches perfectly, then the dimension is extended in order to
    guarantee that.

    The new size of the dimension is calculated by taking the last multiple of stride within the
    dimension and adding the size of a patch to it. With this operation, this extended dimension
    now fits patches perfectly considering their size and stride.

    Args:
        patch_size (int): The size of the patch.
        stride (int): The stride between patches.
        image_shape (Tuple[int, ...]): The shape of the image.

    Returns:
        Tuple[int, ...]: The extended dimensions of the image.

    """
    return [
        (
            image_shape[i]
            # do not extend if patches fit perfectly in the image
            if (image_shape[i] - patch_size) % stride == 0
            # calculate the last multiple of stride + patch_size, which ensures patches fit perfectly.
            else (((image_shape[i] // stride) * stride) + patch_size)
        )
        for i in range(len(image_shape))
    ]

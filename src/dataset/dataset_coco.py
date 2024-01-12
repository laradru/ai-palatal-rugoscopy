# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# A COCO dataset class that should be used with PyTorch.                                                              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.annotations_coco import COCOAnnotations
from src.dataset.annotations_utils import to_dict
from src.dataset.dataset_base import MutableDataset
from src.dataset.dataset_utils import (
    generate_binary_component,
    generate_binary_mask,
    generate_category_mask,
    patch_generator,
    read_image,
)


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

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

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
        all_images = list(self.images.keys())
        total_images = len(all_images)
        subset_sizes = [int(total_images * perc) for perc in percentages]

        if random:
            np.random.shuffle(all_images)

        for ss in subset_sizes:
            subset = deepcopy(self)
            indexes = all_images[:ss]

            subset.tree.data["images"] = [self.images[i][0] for i in indexes]
            subset.images = to_dict(subset.tree.data["images"], "id")

            image_annotations = to_dict(subset.tree.data["annotations"], "image_id")
            subset.tree.data["annotations"] = [image_annotations[image_id][0] for image_id in subset.images.keys()]
            subset.tree.data["annotations"] = np.array(subset.tree.data["annotations"]).flatten().tolist()
            subset.annotations = subset.tree.data.get("annotations")

            subsets.append(subset)
            all_images = all_images[ss:]

        return tuple(subsets)

    def preview_dataset(self) -> None:
        """Prints a preview of the dataset by displaying some dataset statistics."""

        horizontal_bar_length = 100
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
        image_data = self.images[annotation["image_id"]]
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_data = self.images[idx]
        annotations = self.annotations[image_data["id"]]
        image_path = os.path.join(self.data_directory_path, image_data["file_name"])
        image = read_image(image_path)

        # apply preprocessing
        if self.preprocessing is not None:
            image, annotations = self.preprocessing(image, annotations)

        # apply augmentations
        if self.augmentations is not None:
            image, annotations = self.augmentations(image, annotations)

        # Generate instance masks for each annotation
        for annotation in annotations:
            mask = generate_binary_component(image, annotation)
            annotation["mask"] = torch.tensor(mask, dtype=torch.float32)
            annotation["boxes"] = torch.tensor(annotation["bbox"])

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (H, W, C) -> (C, H, W)
        return image, annotations

    def __len__(self) -> int:
        """Returns the length of the object.

        Returns:
            int: The length of the object.
        """

        return len(self.images)

    def extract_patches(self, output_dir: str, patch_size: int, stride: int, min_area: float) -> None:
        """Extracts patches from images and saves them along with their annotations.

        Args:
            output_dir (str): The directory where the patches and annotations will be saved.
            patch_size (int): The size of the patches.
            stride (int): The stride between patches.
            min_area (float): The minimum area required for a patch to be considered.
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

        # Create output subdirectories.
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(annotations_output_dir, exist_ok=True)

        for image_data in tqdm(self.images):
            image_path = os.path.join(self.data_directory_path, image_data["file_name"])
            
            if image_data["id"] not in self.annotations.keys():
                continue
            
            image_annotations = self.annotations[image_data["id"]]

            image = read_image(image_path)

            # ~~ Draw components on masks based on the image annotations made on CVAT.
            binary_mask = generate_binary_mask(image, image_annotations)
            category_mask = generate_category_mask(image, image_annotations)
            __, component_mask = cv2.connectedComponents(binary_mask)

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
                    if any([component_size < min_area for component_size in np.bincount(patch_map.flatten())[1:]]):
                        continue

                    patch_annotations.add_image_instance(image_id, patch_name, patch_rows, patch_cols)
                    patch_category_mask = category_mask[
                        coord[0] : coord[0] + patch_rows, coord[1] : coord[1] + patch_cols
                    ]
                    patch_image = image[coord[0] : coord[0] + patch_rows, coord[1] : coord[1] + patch_cols]

                    # Extract data for each component and create a new annotation instance.
                    for label in np.unique(patch_map)[1:]:  # 0 is background
                        instance_map = np.array(patch_map == label, dtype=np.uint8)
                        instance_category = patch_category_mask[patch_map == label].max()  # It could also be min().
                        instance_contours, _ = cv2.findContours(instance_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        instance_bbox = list(cv2.boundingRect(instance_contours[0]))

                        # Now, organize instance segmentation into a list of [x1, y1, x2, y2, x3, y3, ...]
                        instance_contours = instance_contours[0].reshape(-1, 2).astype(float)
                        instance_segmentation = [0] * (instance_contours.shape[0] * instance_contours.shape[1])
                        instance_segmentation[::2] = instance_contours[:, 0]
                        instance_segmentation[1::2] = instance_contours[:, 1]

                        patch_annotations.add_annotation_instance(
                            id=annotation_id,
                            image_id=image_id,
                            category_id=int(instance_category),
                            bbox=instance_bbox,
                            segmentation=[instance_segmentation],
                        )

                        annotation_id += 1

                    cv2.imwrite(os.path.join(output_dir, "images", patch_name), patch_image)
                    image_id += 1

            except (RuntimeError, StopIteration):
                pass

        patch_annotations.save(output_path=os.path.join(output_dir, "annotations", "annotations.json"))

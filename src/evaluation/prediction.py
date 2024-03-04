from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from src.architectures.segmenter_maskrcnn import MaskRCNNSegmenter
from src.dataset.dataset_utils import extract_bbox_segmentation, join_patches, patch_generator


class MaskRCNNPrediction:

    def __init__(self, model_path: str, num_classes: int, patch_size: int, stride: int, device: str = "cuda:0") -> None:
        """Initializes the MaskRCNNSegmenter.

        Args:
            model_path (str): The path to the model.
            num_classes (int): The number of classes for the model.
            patch_size (int): The size of the patch.
            stride (int): The stride value.
            device (str, optional): The device to be used for computation (default is "cuda:0").
        """

        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.num_classes = num_classes
        self.model = MaskRCNNSegmenter(model_path, num_classes)

        self.model.load()
        self.model.to(self.device)

    def predict_batch(self, batch: torch.Tensor) -> Dict:
        """Predicts the output for a given patch using the trained model.

        Args:
            patch (np.ndarray): The input patch for prediction.
        Returns:
            Dict: The predictions for the input patch.
        """

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(batch, None)

        return predictions

    def preprocess_batch(self, batch: List[np.ndarray]) -> torch.Tensor:
        """Preprocesses the input patch by transposing its dimensions and converting it to a torch Tensor.

        Args:
            patch (np.ndarray): The input patch to be preprocessed.
        Returns:
            torch.Tensor: The preprocessed patch as a torch Tensor.
        """
        batch = np.array(batch).transpose(0, 3, 1, 2)
        return torch.Tensor(batch).to(self.device)

    def postprocess_batch(self, batch_preds: List[Dict], threshold: float = 0.5) -> Tuple[List, List, List]:
        """Generate postprocessed masks, labels, and scores from the prediction.

        Args:
            pred (Dict): The prediction dictionary containing 'masks', 'labels', and 'scores'.
            threshold (float): The minimum confidence threshold for predictions.

        Returns:
            Tuple[List, List, List]: A tuple containing lists of postprocessed masks, labels, and scores.
        """

        masks, labels, scores = [], [], []

        for batch_item in batch_preds:
            masks_i, labels_i, scores_i = batch_item["masks"], batch_item["labels"], batch_item["scores"]
            valid_items = scores_i > threshold
            masks_i, labels_i, scores_i = masks_i[valid_items], labels_i[valid_items], scores_i[valid_items]

            masks.append(masks_i.detach().cpu().numpy().astype(np.uint8).squeeze(1))
            labels.append(labels_i.detach().cpu().tolist())
            scores.append(scores_i.detach().cpu().tolist())

        return masks, labels, scores

    def predict_image(self, image: np.ndarray, batch_size: int, threshold: float = 0.5) -> Dict:
        """Predicts the labels and masks for an input image using the given threshold.

        Args:
            image (np.ndarray): The input image for prediction.
            threshold (float, optional): The confidence threshold for label prediction. Defaults to 0.5.
        Returns:
            Dict: A dictionary containing segmented masks for each category.
        """

        pg = patch_generator(image, self.patch_size, self.stride)
        masks, labels, scores, coords = [], [], [], []

        def batch_prediction(batch_patch: List[np.ndarray], batch_coord: List[Tuple]):
            batch_patch = self.preprocess_batch(batch_patch)
            batch_pred = self.predict_batch(batch_patch)
            patch_masks, patch_labels, patch_scores = self.postprocess_batch(batch_pred, threshold)

            masks.extend(patch_masks)
            labels.extend(patch_labels)
            scores.extend(patch_scores)
            coords.extend(batch_coord)

        try:
            batch_patch, batch_coord = [], []

            while patch_data := next(pg):
                patch, coord = patch_data
                batch_patch.append(patch)
                batch_coord.append(coord)

                if len(batch_patch) == batch_size:
                    batch_prediction(batch_patch, batch_coord)
                    batch_coord, batch_patch = [], []

        except (RuntimeError, StopIteration):
            # Process the remaining patches
            if len(batch_patch) > 0:
                batch_prediction(batch_patch, batch_coord)

            return self.postprocess_image(masks, labels, coords)

    def postprocess_image(self, patch_masks: List, patch_labels: List, patch_coords: List) -> Dict:
        """Postprocesses the image by creating masks for each category based on the patch masks, labels and coordinates.

        Args:
            patch_masks (List): List of patch masks.
            patch_labels (List): List of patch labels.
            patch_coords (List): List of patch coordinates.

        Returns:
            Dict: A dictionary containing segmented masks for each category.
        """

        zero = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        category_patches = {c: [zero.copy()] * len(patch_coords) for c in range(1, self.num_classes + 1)}
        category_images = {c: [] for c in range(1, self.num_classes + 1)}
        patch_names = [f"patch_{coord[0]}_{coord[1]}.jpg" for coord in patch_coords]

        # Create mask for each category
        for patch_id in range(len(patch_masks)):
            for instance_label, instance_mask in zip(patch_labels[patch_id], patch_masks[patch_id]):
                instance_mask = np.round(instance_mask).astype(np.uint8)
                category_patches[instance_label][patch_id] = instance_mask

        for c in range(1, self.num_classes + 1):
            category_images[c] = join_patches(category_patches[c], patch_names)

        return category_images

    @staticmethod
    def to_cvat(category_images: Dict) -> List[Dict]:
        """Convert predictions to a format compatible with CVAT.

        Args:
            category_images (Dict): A dictionary containing category IDs as keys and segmentation maps as values.
        Returns:
            List[Dict]: A list of dictionaries containing confidence, label, points, and type for each category image.
        """

        response = []
        smallest_ob_size = 10

        for category_id, segmentation_map in category_images.items():
            if np.any(segmentation_map > 0):
                __, components_map = cv2.connectedComponents(segmentation_map)
                components_id, counts = np.unique(components_map, return_counts=True)[0:]

                for component_id, count in zip(components_id, counts):
                    if count < smallest_ob_size:
                        continue

                    __, segmentation = extract_bbox_segmentation(components_map == component_id)
                    response.append({"confidence": 1, "label": category_id, "points": segmentation, "type": "polygon"})
        return response

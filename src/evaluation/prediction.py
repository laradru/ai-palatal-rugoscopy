from typing import Dict, List, Tuple

import numpy as np
import torch

from src.architectures.segmenter_maskrcnn import MaskRCNNSegmenter
from src.dataset.dataset_utils import join_patches, patch_generator


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

    def predict_patch(self, patch: np.ndarray) -> Dict:
        """Predicts the output for a given patch using the trained model.

        Args:
            patch (np.ndarray): The input patch for prediction.
        Returns:
            Dict: The predictions for the input patch.
        """

        self.model.eval()
        with torch.no_grad():
            predictions = self.model([patch], None)[0]

        return predictions

    def preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """Preprocesses the input patch by transposing its dimensions and converting it to a torch Tensor.

        Args:
            patch (np.ndarray): The input patch to be preprocessed.
        Returns:
            torch.Tensor: The preprocessed patch as a torch Tensor.
        """

        patch = patch.transpose(2, 0, 1)
        return torch.Tensor(patch).to(self.device)

    def postprocess_patch(self, pred: Dict, threshold: float = 0.5) -> Tuple[List, List, List]:
        """Generate postprocessed masks, labels, and scores from the prediction.

        Args:
            pred (Dict): The prediction dictionary containing 'masks', 'labels', and 'scores'.
            threshold (float): The minimum confidence threshold for predictions.

        Returns:
            Tuple[List, List, List]: A tuple containing lists of postprocessed masks, labels, and scores.
        """

        masks, labels, scores = [], [], []

        for mask, label, score in zip(pred["masks"], pred["labels"], pred["scores"]):
            if score < threshold:
                continue

            mask = mask.detach().cpu().numpy().astype(np.uint8)
            mask = mask.squeeze()
            masks.append(mask)
            labels.append(label.detach().cpu().tolist())
            scores.append(score.detach().cpu().tolist())

        return masks, labels, scores

    def predict_image(self, image: np.ndarray, threshold: float = 0.5) -> Dict:
        """Predicts the labels and masks for an input image using the given threshold.

        Args:
            image (np.ndarray): The input image for prediction.
            threshold (float, optional): The confidence threshold for label prediction. Defaults to 0.5.
        Returns:
            Dict: A dictionary containing segmented masks for each category.
        """

        pg = patch_generator(image, self.patch_size, self.stride)
        masks, labels, scores, coords = [], [], [], []

        try:
            while patch_data := next(pg):
                patch, coord = patch_data
                patch = self.preprocess_patch(patch)
                pred = self.predict_patch(patch)
                patch_masks, patch_labels, patch_scores = self.postprocess_patch(pred, threshold)

                masks.append(patch_masks)
                labels.append(patch_labels)
                scores.append(patch_scores)
                coords.append(coord)

        except (RuntimeError, StopIteration):
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

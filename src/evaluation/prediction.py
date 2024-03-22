from abc import ABC
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T

from src.architectures.arch_base import ArchBase
from src.architectures.segmenter_maskrcnn import MaskRCNNSegmenter
from src.dataset.composer import OrderedCompose


class BasePrediction(ABC):
    def __init__(self, model: ArchBase, device: str = "cuda:0") -> None:
        """Initialize base class.

        Args:
            model (ArchBase): The model to be initialized.
            device (str): The device to be used for initialization. Defaults to "cuda:0".
        """

        self.device = device
        self.model = model
        self.model.eval()

    def predict(self, batch: torch.Tensor) -> Dict:
        """Predicts the output for a given input batch using the trained model.

        Args:
            batch (torch.Tensor): The input batch to make predictions on.
        Returns:
            Dict: A dictionary containing the predictions made by the model.
        """

        with torch.no_grad():
            predictions = self.model([batch], None)[0]
        return predictions


class MaskRCNNPrediction(BasePrediction):

    def __init__(self, model: MaskRCNNSegmenter, preprocs: OrderedCompose, device: str = "cuda:0") -> None:
        """Initializes the MaskRCNNSegmenter.

        Args:
            model_path (str): The path to the model.
            num_classes (int): The number of classes for the model.
            pre_procs (OrderedCompose): A preprocessing pipeline.
            device (str, optional): The device to be used for computation (default is "cuda:0").
        """

        self.preproc_funcs = preprocs

        super().__init__(model, device)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocesses the input patch by transposing its dimensions and converting it to a torch Tensor.

        Args:
            patch (np.ndarray): The input patch to be preprocessed.
        Returns:
            torch.Tensor: The preprocessed patch as a torch Tensor.
        """

        if self.preproc_funcs is not None:
            image, __ = self.preproc_funcs(image, None)

        return T.ToTensor()(image).to(self.device)

    def postprocess(self, pred: List[Dict], threshold: float = 0.5) -> Tuple[List, List, List]:
        """Generate postprocessed masks, labels, and scores from the prediction.

        Args:
            pred (Dict): The prediction dictionary containing 'masks', 'labels', and 'scores'.
            threshold (float): The minimum confidence threshold for predictions. 0 for all predictions.

        Returns:
            Tuple[List, List, List]: A tuple containing lists of postprocessed masks, labels, and scores.
        """

        masks, labels, scores = pred["masks"], pred["labels"], pred["scores"]

        valid_items = scores > threshold
        masks, labels, scores = masks[valid_items], labels[valid_items], scores[valid_items]

        masks = masks.detach().cpu().numpy().squeeze(1)
        labels = labels.detach().cpu().tolist()
        scores = scores.detach().cpu().tolist()

        masks[masks >= 0.5] = 1
        masks[masks < 0.5] = 0
        masks = masks.astype(np.uint8)

        return masks, labels, scores

    def predict_image(self, image: np.ndarray, threshold: float = 0.5) -> Dict:
        """Predicts the labels and masks for an input image using the given threshold.

        Args:
            image (np.ndarray): The input image for prediction.
            threshold (float, optional): The confidence threshold for label prediction. Defaults to 0.5.
        Returns:
            Dict: A dictionary containing segmented masks for each category.
        """

        masks, labels, scores = [], [], []
        batch = self.preprocess(image)  # batch of a single image.

        batch_pred = self.predict(batch)
        masks, labels, scores = self.postprocess(batch_pred, threshold)

        return {"masks": masks, "labels": labels, "scores": scores}

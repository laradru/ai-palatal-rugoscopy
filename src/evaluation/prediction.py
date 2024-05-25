from abc import ABC
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T

from src.architectures.arch_base import ArchBase
from src.architectures.segmenter_maskrcnn import MaskRCNNSegmenter
from src.dataset.annotations_utils import filter_to_single_blob, really_agnostic_segmentation_nms
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


class MaskRCNNPrediction(BasePrediction):

    def __init__(self, model: MaskRCNNSegmenter, preprocs: OrderedCompose, device: str = "cuda:0") -> None:
        """Initializes the MaskRCNNSegmenter.

        Args:
            model_path (str): The path to the model.
            pre_procs (OrderedCompose): A preprocessing pipeline.
            device (str, optional): The device to be used for computation (default is "cuda:0").
        """

        self.preproc_funcs = preprocs

        super().__init__(model, device)

    def predict(self, batch: torch.Tensor) -> Dict:
        """A function that makes predictions using the model.

        Args:
            batch (torch.Tensor): The input batch of data.

        Returns:
            Dict: The predictions made by the model.
        """

        with torch.no_grad():
            predictions = self.model([batch], None)[0]
        return predictions

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocesses the input image using the specified preproc_funcs and converts it to a torch.Tensor.

        Args:
            image (np.ndarray): The input image to be preprocessed.

        Returns:
            torch.Tensor: The preprocessed image as a torch.Tensor.
        """

        if self.preproc_funcs is not None:
            image, __ = self.preproc_funcs(image, None)

        return T.ToTensor()(image).to(self.device)

    def postprocess(
        self,
        pred: List[Dict],
        confidence_threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ) -> Tuple[List, List, List]:
        """Generate postprocessed masks, labels, and scores from the prediction.

        Args:
            pred (Dict): The prediction dictionary containing 'masks', 'labels', and 'scores'.
            confidence_threshold (float): The minimum confidence threshold for predictions. Defaults to 0.5.
            segmentation_threshold (float): The minimum segmentation threshold for predictions. Defaults to 0.5.
            nms_threshold (float): The NMS threshold. Defaults to 0.3.

        Returns:
            Tuple[List, List, List]: A tuple containing lists of postprocessed masks, labels, and scores.
        """

        masks, labels, scores = pred["masks"], pred["labels"], pred["scores"]

        masks = masks.detach().cpu().numpy().squeeze(1)
        labels = labels.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

        # Apply nms algorithm
        valid_items = really_agnostic_segmentation_nms(masks, scores, nms_threshold)
        masks, labels, scores = masks[valid_items], labels[valid_items], scores[valid_items]

        # Filter results by confidence score
        valid_items = scores >= confidence_threshold
        masks, labels, scores = masks[valid_items], labels[valid_items], scores[valid_items]

        masks[masks >= segmentation_threshold] = 1
        masks[masks < segmentation_threshold] = 0
        masks = masks.astype(np.uint8)
        masks = filter_to_single_blob(masks)

        return masks, labels.tolist(), scores.tolist()

    def predict_image(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ) -> Dict:
        """Predicts the labels and masks for an input image using the given threshold.

        Args:
            image (np.ndarray): The input image for prediction.
            confidence_threshold (float, optional): Minimum confidence threshold for predictions. Defaults to 0.5.
            segmentation_threshold (float, optional): Minimum segmentation threshold for predictions. Defaults to 0.5.
            nms_threshold (float, optional): The NMS threshold. Defaults to 0.3.
            tiny_blobs_threshold (float, optional): The threshold for filtering tiny blobs. Defaults to 0.5.

        Returns:
            Dict: A dictionary containing segmented masks for each category.
        """

        masks, labels, scores = [], [], []
        one_batch_image = self.preprocess(image)  # batch of a single image.
        one_batch_image_pred = self.predict(one_batch_image)

        masks, labels, scores = self.postprocess(
            pred=one_batch_image_pred,
            confidence_threshold=confidence_threshold,
            segmentation_threshold=segmentation_threshold,
            nms_threshold=nms_threshold,
        )

        return {"masks": masks, "labels": labels, "scores": scores}

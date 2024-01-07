# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Mask R-CNN object detector                                                                                          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import torch
import torchvision

from src.architectures.arch_base import ArchBase


class MaskRCNNSegmenter(ArchBase):
    def __init__(self, model_path: str, num_classes: int) -> None:
        """Class constructor.

        Args:
            model_path (str): Path to the model checkpoint.
            num_classes (int): Number of classes.
        """

        super().__init__(model_path)

        weights_base = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn(weights=weights_base)

        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden = 256

        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)
        mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden, num_classes)
        self.model.roi_heads.box_predictor = box_predictor
        self.model.roi_heads.mask_predictor = mask_predictor

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.
        Returns:
            torch.Tensor: Output tensor.
        """

        return self.model.forward(x, targets)

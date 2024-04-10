# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Mask R-CNN object detector                                                                                          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import torch
from torchvision.models import detection

from src.architectures.arch_base import ArchBase


class MaskRCNNSegmenter(ArchBase):
    def __init__(self, model_path: str, num_classes: int, **kwargs) -> None:
        """Class constructor.

        Args:
            model_path (str): Path to the model checkpoint.
            num_classes (int): Number of classes.
        """

        super().__init__(model_path)

        self.num_classes = num_classes
        weights_base = detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = detection.mask_rcnn.maskrcnn_resnet50_fpn(weights=weights_base)

        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden = 256

        box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features_box, self.num_classes)
        mask_predictor = detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden, self.num_classes)
        self.model.roi_heads.box_predictor = box_predictor
        self.model.roi_heads.mask_predictor = mask_predictor

        self.learning_rate = kwargs.get("lr", 0.001)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.
        Returns:
            torch.Tensor: Output tensor.
        """

        return self.model.forward(x, targets)

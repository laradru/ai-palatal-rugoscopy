# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Base routines for all deep learning models                                                                          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from abc import ABC

import torch


class ArchBase(torch.nn.Module, ABC):
    def __init__(self, model_path: str) -> None:
        """Class constructor.

        Args:
            model_path (str): The path to the model checkpoint file.
        """

        super().__init__()
        self.model_path = model_path
        self.model: torch.nn.Module = None
        self.optimizer: torch.optim.Optimizer = None

    def save(self) -> bool:
        """Saves the model checkpoint to a file.

        Returns:
            bool: True if the model is successfully saved, False otherwise.
        """

        try:
            if self.model_path is not None:
                state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
                torch.save(state, self.model_path)

                return True
            else:
                return False

        except Exception as excpt:
            print(excpt)
            return False

    def load(self) -> bool:
        """Loads the model from the checkpoint file (weights).

        Returns:
            bool: True if the model was successfully loaded, False otherwise.
        """

        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path)
                self.model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

                return True
            else:
                return False

        except Exception as excpt:
            print(f"Error while loading model checkpoints: {excpt}")
            return False

    def freeze_layer(self, layer_name: str) -> None:
        """Freeze the weights of the specified layer.

        Args:
            layer_name (str): The name of the layer to freeze.
        """

        for param in getattr(self.model, layer_name).parameters():
            param.requires_grad = False

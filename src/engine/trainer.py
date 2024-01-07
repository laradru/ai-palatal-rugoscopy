# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Core module for training a deep learning model for computer vision tasks                                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import torch
from torch.nn.modules import loss
from torch.optim import Optimizer

from src.dataset.dataset_base import BaseDataset
from src.training.tensorboard import TrainingRecorder


class SupervisedTrainer:
    def __init__(self, device: str, model: torch.nn.Module, recorder: TrainingRecorder = None, seed: int = None):
        """Class constructor. Initializes the trainer module for supervised learning.

        Args:
            device (str): The device to use for training (e.g., 'cuda' or 'cpu').
            model (torch.nn.Module): The model to be trained.
            recorder (TrainingRecorder, optional): A training recorder to track training progress. Defaults to None.
            seed (int, optional): The seed to use for reproducibility. Defaults to None.
        """

        self.device = device
        self.model = model
        self.recorder = recorder  # Tensorboard recorder to track training progress.
        self.best_loss = 1e20  # Set to a large value, so that the first validation loss is always better.

        if seed is not None:
            torch.manual_seed(seed)  # CPU
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)  # GPU

        self.model.to(self.device)  # Load model in the GPU

    def train(self, dataset: BaseDataset, optimizer: Optimizer, loss_func: loss) -> float:
        """Trains the model on a given dataset.

        Args:
            dataset (BaseDataset): The dataset to train the model on.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            loss_func (torch.nn.modules.loss): The loss function used for training.

        Returns:
            float: The average training loss over all batches.
        """

        learned_images = 0
        loss_training = {
            "loss_classifier": 0,
            "loss_mask": 0,
            "loss_box_reg": 0,
            "loss_objectness": 0,
            "loss_rpn_box_reg": 0,
        }

        # Set module status to training. Implemented in torch.nn.Module
        self.model.train()

        with torch.set_grad_enabled(True):
            for n, batch in enumerate(dataset):
                x_pred, y_true = batch
                x_pred = [x.to(self.device) for x in x_pred]
                y_true = [{key: yt[key].to(self.device) for key in yt.keys()} for yt in y_true]

                # Zero gradients for each batch
                optimizer.zero_grad()

                # Predict
                losses = self.model(x_pred, y_true)

                # Loss computation and weights correction
                loss = sum(loss for loss in losses.values())
                loss.backward()  # backpropagation
                optimizer.step()

                loss_training = {key: loss_training[key] + value for key, value in losses.items()}
                learned_images += len(x_pred)
        return {key: loss_training[key] / learned_images for key in loss_training.keys()}

    def evaluate(self, dataset: BaseDataset, loss_func: loss) -> float:
        """Calculate the evaluation loss on the given dataset.

        Args:
            dataset (BaseDataset): The dataset to evaluate the model on.
            loss_func (torch.nn.modules.loss): The loss function to calculate the loss.

        Returns:
            float: The average validation loss.
        """

        images_evaluated = 0
        loss_validation = {
            "loss_classifier": 0,
            "loss_mask": 0,
            "loss_box_reg": 0,
            "loss_objectness": 0,
            "loss_rpn_box_reg": 0,
        }

        # Set module status to evalutation. Implemented in torch.nn.Module
        self.model.eval()

        with torch.no_grad():
            for n, batch in enumerate(dataset):
                x_pred, y_true = batch
                x_pred = [x.to(self.device) for x in x_pred]
                y_true = [{key: yt[key].to(self.device) for key in yt.keys()} for yt in y_true]

                # Predict
                losses = self.model(x_pred, y_true)
                loss_validation = {key: loss_validation[key] + value for key, value in losses.items()}

                images_evaluated += len(x_pred)
        return {key: loss_validation[key] / images_evaluated for key in loss_validation.keys()}

    def fit(
        self,
        training_dataset: BaseDataset,
        validation_dataset: BaseDataset,
        optimizer: Optimizer,
        train_loss: loss,
        validation_loss: loss,
        epochs: int,
    ):
        """Fits the model to the training dataset and validates it on the validation dataset for a
        specified number of epochs.

        Parameters:
            training_dataset (BaseDataset): The dataset used for training.
            validation_dataset (BaseDataset): The dataset used for validation.
            optimizer (Optimizer): The optimizer used for training.
            train_loss (loss): The loss function used for training.
            validation_loss (loss): The loss function used for validation.
            epochs (int): The number of epochs to train the model.
        """

        for epoch in range(epochs):
            print(f"Epoch {epoch}")

            loss_training = self.train(training_dataset, optimizer, train_loss)
            loss_validation = self.evaluate(validation_dataset, validation_loss)

            print(f"Loss training: {loss_training}")
            print(f"Loss validation: {loss_validation}")

            if self.recorder:
                self.recorder.record_scalar("training loss", loss_training, epoch)
                self.recorder.record_scalar("validation loss", loss_validation, epoch)

            # Save checkpoint.
            if loss_validation < self.best_loss:
                self.best_loss = loss_validation
                self.model.save()

        if self.recorder:
            self.recorder.close()

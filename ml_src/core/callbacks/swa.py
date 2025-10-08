"""Stochastic Weight Averaging (SWA) callback for improved generalization."""

import torch
from loguru import logger

from ml_src.core.callbacks.base import Callback


class StochasticWeightAveraging(Callback):
    """
    Stochastic Weight Averaging for improved model generalization.

    SWA maintains a running average of model weights during training, typically starting
    after a certain number of epochs. Often improves test accuracy by 0.5-2% with minimal
    additional cost. Used in many SOTA models (YOLO, Stable Diffusion, etc.).

    The SWA model is evaluated separately during validation and metrics are logged to
    TensorBoard with '_swa' suffix.

    Attributes:
        swa_start_epoch: Epoch to start SWA averaging (default: 75% of total epochs)
        swa_lr: Learning rate for SWA phase (default: None, uses current LR)
        annealing_epochs: Number of epochs for LR annealing (default: 10)
        annealing_strategy: Annealing strategy ('cos' or 'linear', default: 'cos')
        swa_model: Averaged model (torch.optim.swa_utils.AveragedModel)
        swa_scheduler: SWA learning rate scheduler

    Example:
        >>> # In config.yaml:
        >>> # training:
        >>> #   callbacks:
        >>> #     - type: 'swa'
        >>> #       swa_start_epoch: 75
        >>> #       swa_lr: 0.0005
        >>> #       annealing_epochs: 10
        >>> #       annealing_strategy: 'cos'
        >>>
        >>> from ml_src.core.callbacks import get_callbacks
        >>> callbacks = get_callbacks(config)
        >>> trainer = get_trainer(..., callbacks=callbacks)
        >>> trainer.train()  # SWA applied automatically after epoch 75
    """

    def __init__(
        self,
        swa_start_epoch=None,
        swa_lr=None,
        annealing_epochs=10,
        annealing_strategy="cos",
    ):
        """
        Initialize SWA callback.

        Args:
            swa_start_epoch: Epoch to start SWA (None = 75% of total epochs)
            swa_lr: Learning rate for SWA phase (None = use current LR)
            annealing_epochs: Number of epochs for LR annealing
            annealing_strategy: 'cos' or 'linear' annealing
        """
        super().__init__()
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.annealing_epochs = annealing_epochs
        self.annealing_strategy = annealing_strategy
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_started = False

        # Validate annealing strategy
        if annealing_strategy not in ["cos", "linear"]:
            raise ValueError(f"annealing_strategy must be 'cos' or 'linear', got '{annealing_strategy}'")

    def on_train_begin(self, trainer):
        """
        Initialize SWA model and scheduler.

        Args:
            trainer: The trainer instance
        """
        # Determine SWA start epoch if not specified
        if self.swa_start_epoch is None:
            # Default: start at 75% of training
            self.swa_start_epoch = int(0.75 * trainer.num_epochs)

        # Initialize SWA model
        self.swa_model = torch.optim.swa_utils.AveragedModel(trainer.model)

        # Initialize SWA scheduler
        if self.swa_lr is None:
            # Use current learning rate
            self.swa_lr = trainer.optimizer.param_groups[0]["lr"]

        self.swa_scheduler = torch.optim.swa_utils.SWALR(
            trainer.optimizer,
            swa_lr=self.swa_lr,
            anneal_epochs=self.annealing_epochs,
            anneal_strategy=self.annealing_strategy,
        )

        logger.info(
            f"SWA initialized: start_epoch={self.swa_start_epoch}, "
            f"swa_lr={self.swa_lr}, annealing={self.annealing_epochs} epochs ({self.annealing_strategy})"
        )

    def on_epoch_end(self, trainer, epoch, logs):
        """
        Update SWA model and evaluate if in SWA phase.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        # Check if we should start SWA
        if epoch >= self.swa_start_epoch:
            if not self.swa_started:
                logger.opt(colors=True).info(
                    f"<green>Starting Stochastic Weight Averaging at epoch {epoch}</green>"
                )
                self.swa_started = True

            # Update SWA model
            self.swa_model.update_parameters(trainer.model)

            # Step SWA scheduler
            self.swa_scheduler.step()

            # Evaluate SWA model on validation set
            self._evaluate_swa_model(trainer, epoch)

    def _evaluate_swa_model(self, trainer, epoch):
        """
        Evaluate SWA model on validation set and log metrics.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
        """
        self.swa_model.eval()
        swa_running_loss = 0.0
        swa_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in trainer.dataloaders["val"]:
                inputs = inputs.to(trainer.device)
                labels = labels.to(trainer.device)

                # Forward pass with SWA model
                outputs = self.swa_model(inputs)
                loss = trainer.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                swa_running_loss += loss.item() * inputs.size(0)
                swa_running_corrects += torch.sum(preds == labels.data)

        # Compute metrics
        swa_epoch_loss = swa_running_loss / trainer.dataset_sizes["val"]
        swa_epoch_acc = swa_running_corrects.double() / trainer.dataset_sizes["val"]

        # Log to TensorBoard
        trainer.writer.add_scalar("Loss/val_swa", swa_epoch_loss, epoch)
        trainer.writer.add_scalar("Accuracy/val_swa", swa_epoch_acc.item(), epoch)

        # Log to console
        logger.opt(colors=True).info(
            f"<cyan>val_swa</> Loss: <yellow>{swa_epoch_loss:.4f}</> "
            f"Acc: <yellow>{swa_epoch_acc:.4f}</>"
        )

    def on_train_end(self, trainer):
        """
        Update BatchNorm statistics for SWA model at end of training.

        Args:
            trainer: The trainer instance
        """
        if self.swa_started:
            logger.info("Updating SWA model BatchNorm statistics...")

            # Update BatchNorm statistics
            torch.optim.swa_utils.update_bn(
                trainer.dataloaders["train"],
                self.swa_model,
                device=trainer.device,
            )

            # Save SWA model
            swa_path = trainer.best_model_path.replace("best.pt", "swa.pt")
            torch.save(
                {
                    "model_state_dict": self.swa_model.module.state_dict(),
                    "config": trainer.config,
                },
                swa_path,
            )
            logger.success(f"SWA model saved to {swa_path}")

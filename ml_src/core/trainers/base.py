"""Base trainer abstract class for training models."""

import os
import time
from abc import ABC, abstractmethod

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from ml_src.core.checkpointing import count_parameters, save_summary
from ml_src.core.early_stopping import EarlyStopping
from ml_src.core.metrics import (
    get_classification_report_str,
    log_confusion_matrix_to_tensorboard,
    save_classification_report,
)


def collect_predictions(model, dataloader, device):
    """
    Collect all predictions and labels from a dataloader.

    Args:
        model: The model to use
        dataloader: DataLoader to iterate through
        device: Device to run on

    Returns:
        Tuple of (all_labels, all_predictions)

    Example:
        >>> labels, preds = collect_predictions(model, val_loader, device)
        >>> accuracy = (labels == preds).sum() / len(labels)
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds


class BaseTrainer(ABC):
    """
    Abstract base class for training models.

    This class defines the common training loop structure and provides abstract methods
    that subclasses must implement for specific training strategies (standard PyTorch,
    mixed precision, distributed training, etc.).

    Attributes:
        model: The model to train
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        dataloaders: Dictionary of dataloaders for train and val
        dataset_sizes: Dictionary of dataset sizes
        device: Device to train on
        config: Configuration dictionary
        run_dir: Directory to save model checkpoints
        class_names: List of class names
        writer: TensorBoard writer
        num_epochs: Number of epochs to train
        early_stopping: EarlyStopping instance (None if disabled)

    Example:
        >>> class StandardTrainer(BaseTrainer):
        ...     def prepare_training(self):
        ...         pass
        ...     def training_step(self, inputs, labels):
        ...         return outputs, loss
        ...     # ... implement other abstract methods
        >>> trainer = StandardTrainer(model, criterion, optimizer, ...)
        >>> trained_model, losses, accs = trainer.train()
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        device,
        config,
        run_dir,
        class_names,
    ):
        """
        Initialize the base trainer.

        Args:
            model: The model to train
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            dataloaders: Dictionary of dataloaders for train and val
            dataset_sizes: Dictionary of dataset sizes
            device: Device to train on
            config: Configuration dictionary
            run_dir: Directory to save model checkpoints
            class_names: List of class names
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.config = config
        self.run_dir = run_dir
        self.class_names = class_names

        # Extract num_epochs from config
        self.num_epochs = config["training"]["num_epochs"]

        # Paths for checkpoints and summary
        self.best_model_path = os.path.join(run_dir, "weights", "best.pt")
        self.last_model_path = os.path.join(run_dir, "weights", "last.pt")
        self.summary_path = os.path.join(run_dir, "summary.txt")

        # Initialize TensorBoard writer
        tensorboard_dir = os.path.join(run_dir, "tensorboard")
        self.writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard logs: {tensorboard_dir}")

        # Count model parameters
        self.num_params = count_parameters(model)
        logger.info(f"Model has {self.num_params:,} trainable parameters")

        # Initialize early stopping if enabled
        self.early_stopping = None
        if config.get("training", {}).get("early_stopping", {}).get("enabled", False):
            es_config = config["training"]["early_stopping"]
            self.early_stopping = EarlyStopping(
                patience=es_config.get("patience", 10),
                metric=es_config.get("metric", "val_acc"),
                mode=es_config.get("mode", "max"),
                min_delta=es_config.get("min_delta", 0.0),
            )
            logger.info(
                f"Early stopping enabled: patience={self.early_stopping.patience}, "
                f"metric={self.early_stopping.metric}, mode={self.early_stopping.mode}"
            )

        # Optuna trial for pruning (set externally if used in hyperparameter search)
        self.optuna_trial = None

    @abstractmethod
    def prepare_training(self):
        """
        Prepare for training (e.g., move model to device, setup for mixed precision, etc.).

        This method is called once before training begins. Subclasses should implement
        any setup specific to their training strategy.

        Example:
            >>> def prepare_training(self):
            ...     # Standard PyTorch: no special preparation needed
            ...     pass
            >>> def prepare_training(self):
            ...     # Mixed precision: setup GradScaler
            ...     self.scaler = torch.cuda.amp.GradScaler()
        """
        pass

    @abstractmethod
    def training_step(self, inputs, labels):
        """
        Execute a single training step (forward + backward + optimizer step).

        Args:
            inputs: Input batch
            labels: Target labels

        Returns:
            Tuple of (outputs, loss) where:
                - outputs: Model outputs (logits)
                - loss: Computed loss value (tensor)

        Example:
            >>> def training_step(self, inputs, labels):
            ...     outputs = self.model(inputs)
            ...     loss = self.criterion(outputs, labels)
            ...     loss.backward()
            ...     self.optimizer.step()
            ...     return outputs, loss
        """
        pass

    @abstractmethod
    def validation_step(self, inputs, labels):
        """
        Execute a single validation step (forward pass only).

        Args:
            inputs: Input batch
            labels: Target labels

        Returns:
            Tuple of (outputs, loss) where:
                - outputs: Model outputs (logits)
                - loss: Computed loss value (tensor)

        Example:
            >>> def validation_step(self, inputs, labels):
            ...     outputs = self.model(inputs)
            ...     loss = self.criterion(outputs, labels)
            ...     return outputs, loss
        """
        pass

    @abstractmethod
    def save_checkpoint(self, epoch, best_acc, metrics, path):
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch number
            best_acc: Best validation accuracy achieved so far
            metrics: Dictionary containing train_losses, val_losses, train_accs, val_accs
            path: Path to save the checkpoint

        Example:
            >>> def save_checkpoint(self, epoch, best_acc, metrics, path):
            ...     from ml_src.core.checkpointing import save_checkpoint
            ...     save_checkpoint(
            ...         self.model, self.optimizer, self.scheduler, epoch, best_acc,
            ...         metrics['train_losses'], metrics['val_losses'],
            ...         metrics['train_accs'], metrics['val_accs'],
            ...         self.config, path
            ...     )
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        """
        Load a checkpoint for resuming training.

        Args:
            path: Path to the checkpoint file

        Returns:
            Tuple of (epoch, best_acc, train_losses, val_losses, train_accs, val_accs)

        Example:
            >>> def load_checkpoint(self, path):
            ...     from ml_src.core.checkpointing import load_checkpoint
            ...     return load_checkpoint(path, self.model, self.optimizer,
            ...                          self.scheduler, self.device)
        """
        pass

    def train(
        self,
        start_epoch=0,
        resume_best_acc=0.0,
        resume_train_losses=None,
        resume_val_losses=None,
        resume_train_accs=None,
        resume_val_accs=None,
    ):
        """
        Main training loop.

        Args:
            start_epoch: Epoch to start from (for resuming)
            resume_best_acc: Best accuracy from resumed checkpoint
            resume_train_losses: Training losses from resumed checkpoint
            resume_val_losses: Validation losses from resumed checkpoint
            resume_train_accs: Training accuracies from resumed checkpoint
            resume_val_accs: Validation accuracies from resumed checkpoint

        Returns:
            Tuple of (model, train_losses, val_losses, train_accs, val_accs)
        """
        since = time.time()

        # Initialize or resume metrics
        best_acc = resume_best_acc
        best_epoch = start_epoch if resume_best_acc > 0 else 0
        train_losses = resume_train_losses if resume_train_losses is not None else []
        val_losses = resume_val_losses if resume_val_losses is not None else []
        train_accs = resume_train_accs if resume_train_accs is not None else []
        val_accs = resume_val_accs if resume_val_accs is not None else []
        early_stop_triggered = False

        # Prepare for training (subclass-specific setup)
        self.prepare_training()

        # Save initial checkpoint if starting fresh
        if start_epoch == 0:
            metrics = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            }
            self.save_checkpoint(0, best_acc, metrics, self.best_model_path)

        # Create initial summary
        save_summary(
            summary_path=self.summary_path,
            status="running",
            config=self.config,
            device=str(self.device),
            dataset_sizes=self.dataset_sizes,
            num_parameters=self.num_params,
            start_time=since,
            current_epoch=start_epoch,
            total_epochs=self.num_epochs,
            best_acc=best_acc if best_acc > 0 else None,
            best_epoch=best_epoch if best_acc > 0 else None,
        )

        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            logger.opt(colors=True).info(f"<yellow>Epoch {epoch}/{self.num_epochs - 1}</yellow>")
            logger.opt(colors=True).info("<dim>" + "-" * 50 + "</dim>")

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    if phase == "train":
                        outputs, loss = self.training_step(inputs, labels)
                    else:
                        with torch.no_grad():
                            outputs, loss = self.validation_step(inputs, labels)

                    # Get predictions
                    _, preds = torch.max(outputs, 1)

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    self.scheduler.step()
                    # Log learning rate to TensorBoard
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("Learning_Rate", current_lr, epoch)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                phase_color = "<blue>" if phase == "train" else "<magenta>"
                logger.opt(colors=True).info(
                    f"{phase_color}{phase}</> Loss: <yellow>{epoch_loss:.4f}</> "
                    f"Acc: <yellow>{epoch_acc:.4f}</>"
                )

                # Store metrics
                if phase == "train":
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc.item())
                    # Log to TensorBoard
                    self.writer.add_scalar("Loss/train", epoch_loss, epoch)
                    self.writer.add_scalar("Accuracy/train", epoch_acc.item(), epoch)
                else:
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc.item())
                    # Log to TensorBoard
                    self.writer.add_scalar("Loss/val", epoch_loss, epoch)
                    self.writer.add_scalar("Accuracy/val", epoch_acc.item(), epoch)

                # Save best model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    metrics = {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "train_accs": train_accs,
                        "val_accs": val_accs,
                    }
                    self.save_checkpoint(epoch, best_acc, metrics, self.best_model_path)
                    logger.success(f"New best model saved! Acc: {best_acc:.4f}")

                # Report to Optuna trial for pruning (if running hyperparameter search)
                if phase == "val" and self.optuna_trial is not None:
                    try:
                        import optuna

                        # Report intermediate value to Optuna
                        self.optuna_trial.report(epoch_acc.item(), epoch)

                        # Check if trial should be pruned
                        if self.optuna_trial.should_prune():
                            logger.warning(f"Trial pruned at epoch {epoch}")
                            raise optuna.TrialPruned()
                    except ImportError:
                        pass  # Optuna not installed, skip pruning

                # Check early stopping
                if phase == "val" and self.early_stopping is not None:
                    metric_value = (
                        epoch_acc.item() if self.early_stopping.metric == "val_acc" else epoch_loss
                    )
                    if self.early_stopping.should_stop(epoch, metric_value):
                        # Save last checkpoint before stopping
                        metrics = {
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "train_accs": train_accs,
                            "val_accs": val_accs,
                        }
                        self.save_checkpoint(epoch, best_acc, metrics, self.last_model_path)
                        # Set flag to break outer loop after summary update
                        early_stop_triggered = True
                        break

            # Check if we need to break out of training loop
            if early_stop_triggered:
                # Update summary for early stopping
                stopped_epoch = self.early_stopping.stopped_epoch if self.early_stopping else epoch
                save_summary(
                    summary_path=self.summary_path,
                    status=f"early_stopped_epoch_{stopped_epoch}",
                    config=self.config,
                    device=str(self.device),
                    dataset_sizes=self.dataset_sizes,
                    num_parameters=self.num_params,
                    start_time=since,
                    current_epoch=epoch + 1,
                    total_epochs=self.num_epochs,
                    best_acc=best_acc,
                    best_epoch=best_epoch,
                    final_train_acc=train_accs[-1] if train_accs else None,
                    final_train_loss=train_losses[-1] if train_losses else None,
                    final_val_acc=val_accs[-1] if val_accs else None,
                    final_val_loss=val_losses[-1] if val_losses else None,
                )
                break

            # Save last model checkpoint after each epoch
            metrics = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            }
            self.save_checkpoint(epoch, best_acc, metrics, self.last_model_path)

            # Update summary after each epoch
            save_summary(
                summary_path=self.summary_path,
                status="running",
                config=self.config,
                device=str(self.device),
                dataset_sizes=self.dataset_sizes,
                num_parameters=self.num_params,
                start_time=since,
                current_epoch=epoch + 1,
                total_epochs=self.num_epochs,
                best_acc=best_acc,
                best_epoch=best_epoch,
                final_train_acc=train_accs[-1] if train_accs else None,
                final_train_loss=train_losses[-1] if train_losses else None,
                final_val_acc=val_accs[-1] if val_accs else None,
                final_val_loss=val_losses[-1] if val_losses else None,
            )

            logger.info("")

        time_elapsed = time.time() - since
        end_time = time.time()

        # Log completion message
        if early_stop_triggered:
            logger.success(
                f"Training stopped early in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
        else:
            logger.success(
                f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
        logger.success(f"Best val Acc: {best_acc:.4f}")

        # Load best model weights
        checkpoint = torch.load(self.best_model_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Generate metrics for train and val sets
        logger.info("Generating metrics for train and val sets...")

        # Collect predictions for train
        train_labels, train_preds = collect_predictions(
            self.model, self.dataloaders["train"], self.device
        )

        # Log to TensorBoard
        log_confusion_matrix_to_tensorboard(
            self.writer,
            train_labels,
            train_preds,
            self.class_names,
            "Confusion_Matrix/train",
            self.num_epochs - 1,
        )
        train_report = get_classification_report_str(train_labels, train_preds, self.class_names)
        self.writer.add_text("Classification_Report/train", train_report, self.num_epochs - 1)

        # Save to files (for backward compatibility)
        save_classification_report(
            train_labels,
            train_preds,
            self.class_names,
            os.path.join(self.run_dir, "logs", "classification_report_train.txt"),
        )

        # Collect predictions for val
        val_labels, val_preds = collect_predictions(
            self.model, self.dataloaders["val"], self.device
        )

        # Log to TensorBoard
        log_confusion_matrix_to_tensorboard(
            self.writer,
            val_labels,
            val_preds,
            self.class_names,
            "Confusion_Matrix/val",
            self.num_epochs - 1,
        )
        val_report = get_classification_report_str(val_labels, val_preds, self.class_names)
        self.writer.add_text("Classification_Report/val", val_report, self.num_epochs - 1)

        # Save to files (for backward compatibility)
        save_classification_report(
            val_labels,
            val_preds,
            self.class_names,
            os.path.join(self.run_dir, "logs", "classification_report_val.txt"),
        )

        # Save final summary
        if early_stop_triggered:
            stopped_epoch = self.early_stopping.stopped_epoch if self.early_stopping else epoch
            final_status = f"early_stopped_epoch_{stopped_epoch}"
        else:
            final_status = "completed"
        save_summary(
            summary_path=self.summary_path,
            status=final_status,
            config=self.config,
            device=str(self.device),
            dataset_sizes=self.dataset_sizes,
            num_parameters=self.num_params,
            start_time=since,
            end_time=end_time,
            current_epoch=len(train_accs),
            total_epochs=self.num_epochs,
            best_acc=best_acc,
            best_epoch=best_epoch,
            final_train_acc=train_accs[-1] if train_accs else None,
            final_train_loss=train_losses[-1] if train_losses else None,
            final_val_acc=val_accs[-1] if val_accs else None,
            final_val_loss=val_losses[-1] if val_losses else None,
        )

        # Close TensorBoard writer
        self.writer.close()
        logger.info("TensorBoard writer closed")

        return self.model, train_losses, val_losses, train_accs, val_accs

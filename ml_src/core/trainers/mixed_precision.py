"""Mixed precision trainer implementation using PyTorch AMP."""

import torch
from loguru import logger

from ml_src.core.checkpointing import load_checkpoint, save_checkpoint
from ml_src.core.trainers.base import BaseTrainer


class MixedPrecisionTrainer(BaseTrainer):
    """
    Mixed precision trainer using PyTorch Automatic Mixed Precision (AMP).

    This trainer uses torch.cuda.amp to enable mixed precision training:
    - Faster training with reduced memory usage
    - Automatic scaling for gradient stability
    - Support for both float16 and bfloat16 dtypes

    Falls back to standard training on CPU with a warning.

    Example:
        >>> trainer = MixedPrecisionTrainer(
        ...     model=model,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     dataloaders=dataloaders,
        ...     dataset_sizes=dataset_sizes,
        ...     device=device,
        ...     config=config,
        ...     run_dir=run_dir,
        ...     class_names=class_names
        ... )
        >>> model, train_losses, val_losses, train_accs, val_accs = trainer.train()
    """

    def prepare_training(self):
        """
        Prepare for mixed precision training.

        Initializes GradScaler for CUDA training. If device is CPU, logs a warning
        and disables mixed precision (falls back to standard training).
        """
        # Check if device is CUDA
        if self.device.type == "cuda":
            # Initialize GradScaler for mixed precision
            self.scaler = torch.cuda.amp.GradScaler()

            # Get amp_dtype from config (default to float16)
            amp_dtype_str = self.config.get("training", {}).get("amp_dtype", "float16")

            # Map string to torch dtype
            if amp_dtype_str == "bfloat16":
                self.amp_dtype = torch.bfloat16
                dtype_name = "bfloat16"
            else:
                self.amp_dtype = torch.float16
                dtype_name = "float16"

            logger.info(f"Mixed precision training enabled with dtype={dtype_name}")
        else:
            # CPU doesn't support AMP - fall back to standard training
            self.scaler = None
            self.amp_dtype = None
            logger.warning(
                "Mixed precision training requested but device is CPU. "
                "Falling back to standard training (AMP requires CUDA)."
            )

    def training_step(self, inputs, labels):
        """
        Execute a single training step with mixed precision.

        Performs:
        1. Forward pass with autocast (if CUDA)
        2. Loss calculation
        3. Backward pass with gradient scaling (if CUDA)
        4. Optimizer step with scaler (if CUDA)

        Args:
            inputs: Input batch (already on device)
            labels: Target labels (already on device)

        Returns:
            Tuple of (outputs, loss):
                - outputs: Model outputs (logits)
                - loss: Computed loss value (tensor)
        """
        if self.device.type == "cuda" and self.scaler is not None:
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Fall back to standard training on CPU
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Standard backward pass
            loss.backward()

            # Standard optimizer step
            self.optimizer.step()

        return outputs, loss

    def validation_step(self, inputs, labels):
        """
        Execute a single validation step with mixed precision.

        Performs:
        1. Forward pass with autocast (if CUDA, no gradient tracking)
        2. Loss calculation

        Args:
            inputs: Input batch (already on device)
            labels: Target labels (already on device)

        Returns:
            Tuple of (outputs, loss):
                - outputs: Model outputs (logits)
                - loss: Computed loss value (tensor)
        """
        if self.device.type == "cuda" and self.amp_dtype is not None:
            # Mixed precision forward pass (no gradient tracking - handled by BaseTrainer)
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
        else:
            # Fall back to standard forward pass on CPU
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

        return outputs, loss

    def save_checkpoint(self, epoch, best_acc, metrics, path):
        """
        Save a mixed precision checkpoint.

        Saves:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict
        - GradScaler state dict (if using CUDA)
        - Training metrics history
        - Random states for reproducibility
        - Early stopping state (if enabled)

        Args:
            epoch: Current epoch number
            best_acc: Best validation accuracy achieved so far
            metrics: Dictionary containing train_losses, val_losses, train_accs, val_accs
            path: Path to save the checkpoint
        """
        # Get early stopping state if enabled
        early_stopping_state = None
        if self.early_stopping is not None:
            early_stopping_state = self.early_stopping.get_state()

        # Use standard checkpointing function
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_acc=best_acc,
            train_losses=metrics["train_losses"],
            val_losses=metrics["val_losses"],
            train_accs=metrics["train_accs"],
            val_accs=metrics["val_accs"],
            config=self.config,
            checkpoint_path=path,
            early_stopping_state=early_stopping_state,
        )

        # Additionally save GradScaler state if using mixed precision
        if self.device.type == "cuda" and self.scaler is not None:
            # Load the checkpoint we just saved
            checkpoint = torch.load(path, weights_only=False)

            # Add scaler state
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

            # Save again with scaler state
            torch.save(checkpoint, path)
            logger.debug(f"Saved GradScaler state to checkpoint: {path}")

    def load_checkpoint(self, path):
        """
        Load a mixed precision checkpoint.

        Restores:
        - Model weights
        - Optimizer state
        - Scheduler state
        - GradScaler state (if present and using CUDA)
        - Training metrics history
        - Random states for reproducibility

        Args:
            path: Path to the checkpoint file

        Returns:
            Tuple of (epoch, best_acc, train_losses, val_losses, train_accs, val_accs)
        """
        # Load standard checkpoint components
        (
            epoch,
            best_acc,
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            config,
            early_stopping_state,
        ) = load_checkpoint(
            checkpoint_path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        # Restore early stopping state if available
        if early_stopping_state is not None and self.early_stopping is not None:
            self.early_stopping.load_state(early_stopping_state)
            logger.success("Restored early stopping state from checkpoint")

        # Load GradScaler state if present and using CUDA
        if self.device.type == "cuda" and self.scaler is not None:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                logger.debug("Loaded GradScaler state from checkpoint")
            else:
                logger.warning(
                    "Checkpoint does not contain GradScaler state. Starting with fresh GradScaler."
                )

        return epoch, best_acc, train_losses, val_losses, train_accs, val_accs

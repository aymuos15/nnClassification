"""Accelerate-based trainer for multi-GPU and distributed training."""

from accelerate import Accelerator

from ml_src.core.checkpointing import load_checkpoint, save_checkpoint
from ml_src.core.trainers.base import BaseTrainer


class AccelerateTrainer(BaseTrainer):
    """
    HuggingFace Accelerate trainer for multi-GPU and distributed training.

    This trainer uses HuggingFace Accelerate to provide:
    - Seamless multi-GPU training (DataParallel/DistributedDataParallel)
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation
    - Automatic device placement
    - Single-device fallback (works with both `python ml-train` and `accelerate launch`)

    The trainer automatically handles:
    - Model/optimizer/dataloader preparation
    - Gradient accumulation
    - Device placement (no manual .to(device) needed)
    - Multi-process synchronization

    Configuration:
        training:
            trainer_type: 'accelerate'
            gradient_accumulation_steps: 1  # Optional: accumulate gradients over N batches

    Usage:
        Single device:
            python ml-train --config config.yaml

        Multi-GPU (after `accelerate config`):
            accelerate launch ml-train --config config.yaml

    Example:
        >>> trainer = AccelerateTrainer(
        ...     model=model,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     dataloaders=dataloaders,
        ...     dataset_sizes=dataset_sizes,
        ...     device=device,  # Ignored (Accelerator handles device)
        ...     config=config,
        ...     run_dir=run_dir,
        ...     class_names=class_names
        ... )
        >>> model, train_losses, val_losses, train_accs, val_accs = trainer.train()
    """

    def prepare_training(self):
        """
        Prepare for Accelerate training.

        Initializes Accelerator and prepares model, optimizer, and dataloaders.
        The Accelerator handles:
        - Device placement
        - Distributed setup
        - Mixed precision configuration
        - Gradient accumulation

        After this method, all training components are moved to the correct device(s)
        and wrapped for distributed training if applicable.
        """
        # Get gradient accumulation steps from config (default: 1)
        gradient_accumulation_steps = self.config.get("training", {}).get(
            "gradient_accumulation_steps", 1
        )

        # Initialize Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # Prepare model, optimizer, and dataloaders
        # Accelerator handles device placement and distributed wrapping
        self.model, self.optimizer, self.dataloaders["train"], self.dataloaders["val"] = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.dataloaders["train"], self.dataloaders["val"]
            )
        )

        # Update device to match Accelerator's device
        # This ensures BaseTrainer's .to(device) calls in the main loop work correctly
        self.device = self.accelerator.device

        # Log device info (only on main process)
        if self.accelerator.is_main_process:
            from loguru import logger

            logger.info(f"Accelerator device: {self.accelerator.device}")
            logger.info(
                f"Gradient accumulation steps: {self.accelerator.gradient_accumulation_steps}"
            )
            logger.info(f"Distributed type: {self.accelerator.distributed_type}")
            logger.info(f"Number of processes: {self.accelerator.num_processes}")

    def training_step(self, inputs, labels):
        """
        Execute a single training step with Accelerate.

        Uses accelerator.backward() instead of loss.backward() to handle:
        - Gradient accumulation
        - Mixed precision scaling
        - Distributed gradient synchronization

        Args:
            inputs: Input batch (already on correct device via Accelerator)
            labels: Target labels (already on correct device via Accelerator)

        Returns:
            Tuple of (outputs, loss):
                - outputs: Model outputs (logits)
                - loss: Computed loss value (tensor)
        """
        # Forward pass (Accelerator handles device placement)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        # Backward pass using Accelerator
        # This handles gradient accumulation and mixed precision
        self.accelerator.backward(loss)

        # Optimizer step
        self.optimizer.step()

        return outputs, loss

    def validation_step(self, inputs, labels):
        """
        Execute a single validation step with Accelerate.

        Standard forward pass - Accelerator handles device placement automatically.

        Args:
            inputs: Input batch (already on correct device via Accelerator)
            labels: Target labels (already on correct device via Accelerator)

        Returns:
            Tuple of (outputs, loss):
                - outputs: Model outputs (logits)
                - loss: Computed loss value (tensor)
        """
        # Forward pass (Accelerator handles device placement)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        return outputs, loss

    def save_checkpoint(self, epoch, best_acc, metrics, path):
        """
        Save a checkpoint with Accelerate.

        Uses accelerator.unwrap_model() to get the raw model state dict
        (unwrapping any distributed wrappers like DistributedDataParallel).

        Only saves on the main process to avoid race conditions.

        Args:
            epoch: Current epoch number
            best_acc: Best validation accuracy achieved so far
            metrics: Dictionary containing train_losses, val_losses, train_accs, val_accs
            path: Path to save the checkpoint
        """
        # Only save on main process to avoid race conditions
        if self.accelerator.is_main_process:
            # Unwrap model from distributed wrapper before saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            # Get early stopping state if enabled
            early_stopping_state = None
            if self.early_stopping is not None:
                early_stopping_state = self.early_stopping.get_state()

            save_checkpoint(
                model=unwrapped_model,
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

        # Wait for all processes to reach this point
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path):
        """
        Load a checkpoint for resuming training with Accelerate.

        Loads state dict into the unwrapped model, then re-prepares with Accelerator.

        Args:
            path: Path to the checkpoint file

        Returns:
            Tuple of (epoch, best_acc, train_losses, val_losses, train_accs, val_accs)
        """
        from loguru import logger

        # Unwrap model before loading state dict
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        (
            epoch,
            best_acc,
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            _,
            early_stopping_state,
        ) = load_checkpoint(
            checkpoint_path=path,
            model=unwrapped_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.accelerator.device,
        )

        # Restore early stopping state if available
        if early_stopping_state is not None and self.early_stopping is not None:
            self.early_stopping.load_state(early_stopping_state)
            logger.success("Restored early stopping state from checkpoint")

        return epoch, best_acc, train_losses, val_losses, train_accs, val_accs

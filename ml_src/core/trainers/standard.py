"""Standard PyTorch trainer implementation."""


from ml_src.core.checkpointing import load_checkpoint, save_checkpoint
from ml_src.core.trainers.base import BaseTrainer


class StandardTrainer(BaseTrainer):
    """
    Standard PyTorch trainer with manual device management.

    This trainer implements the traditional PyTorch training pattern with:
    - Manual .to(device) for tensors
    - Standard forward/backward passes
    - No mixed precision or distributed training
    - Full callback support

    This is the default trainer and maintains backward compatibility with
    existing training workflows.

    Example:
        >>> trainer = StandardTrainer(
        ...     model=model,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     dataloaders=dataloaders,
        ...     dataset_sizes=dataset_sizes,
        ...     device=device,
        ...     config=config,
        ...     run_dir=run_dir,
        ...     class_names=class_names,
        ...     callbacks=callbacks  # Optional callbacks
        ... )
        >>> model, train_losses, val_losses, train_accs, val_accs = trainer.train()
    """

    def prepare_training(self):
        """
        Prepare for standard PyTorch training.

        For standard training, no special preparation is needed. The model is
        already on the correct device from get_model().
        """
        # No special preparation needed for standard training
        pass

    def training_step(self, inputs, labels):
        """
        Execute a single training step with standard PyTorch.

        Performs:
        1. Forward pass
        2. Loss calculation
        3. Backward pass
        4. Optimizer step

        Args:
            inputs: Input batch (already on device)
            labels: Target labels (already on device)

        Returns:
            Tuple of (outputs, loss):
                - outputs: Model outputs (logits)
                - loss: Computed loss value (tensor)
        """
        # Forward pass with gradient tracking enabled
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        return outputs, loss

    def validation_step(self, inputs, labels):
        """
        Execute a single validation step with standard PyTorch.

        Performs:
        1. Forward pass (no gradient tracking)
        2. Loss calculation

        Args:
            inputs: Input batch (already on device)
            labels: Target labels (already on device)

        Returns:
            Tuple of (outputs, loss):
                - outputs: Model outputs (logits)
                - loss: Computed loss value (tensor)
        """
        # Forward pass (no gradient tracking - handled by BaseTrainer)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        return outputs, loss

    def save_checkpoint(self, epoch, best_acc, metrics, path):
        """
        Save a standard PyTorch checkpoint.

        Uses the existing checkpointing module to save:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict
        - Training metrics history
        - Random states for reproducibility
        - Early stopping state (if enabled)
        - EMA state (if enabled)

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

        # Get EMA state if enabled
        ema_state = None
        if self.ema is not None:
            ema_state = self.ema.state_dict()

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
            ema_state=ema_state,
        )

    def load_checkpoint(self, path):
        """
        Load a standard PyTorch checkpoint.

        Restores:
        - Model weights
        - Optimizer state
        - Scheduler state
        - Training metrics history
        - Random states for reproducibility
        - Early stopping state (if available)
        - EMA state (if available)

        Args:
            path: Path to the checkpoint file

        Returns:
            Tuple of (epoch, best_acc, train_losses, val_losses, train_accs, val_accs)
        """
        from loguru import logger

        (
            epoch,
            best_acc,
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            _,
            early_stopping_state,
            ema_state,
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

        # Restore EMA state if available
        if ema_state is not None and self.ema is not None:
            self.ema.load_state_dict(ema_state)
            logger.success("Restored EMA state from checkpoint")

        return epoch, best_acc, train_losses, val_losses, train_accs, val_accs

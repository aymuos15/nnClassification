"""Base callback classes for training lifecycle hooks."""


class Callback:
    """
    Base callback class for injecting custom behavior into the training loop.

    Callbacks provide hooks at various points in the training lifecycle, allowing users to
    implement custom logic without modifying the core trainer code. All hook methods receive
    the trainer instance, providing access to the model, optimizer, data, and other state.

    Lifecycle order:
        on_train_begin
          └─ [for each epoch]
              ├─ on_epoch_begin
              ├─ on_phase_begin (phase='train')
              │   └─ [for each batch]
              │       ├─ on_batch_begin
              │       ├─ on_backward_begin
              │       ├─ on_backward_end
              │       ├─ on_optimizer_step_begin
              │       ├─ on_optimizer_step_end
              │       └─ on_batch_end
              ├─ on_phase_end (phase='train', logs={...})
              ├─ on_phase_begin (phase='val')
              │   └─ [validation batches...]
              ├─ on_phase_end (phase='val', logs={...})
              └─ on_epoch_end (logs={train_loss, val_loss, train_acc, val_acc, lr, ...})
        on_train_end

    Attributes:
        None by default. Subclasses can define their own attributes.

    Example:
        >>> class PrintEpochCallback(Callback):
        ...     def on_epoch_end(self, trainer, epoch, logs):
        ...         print(f"Epoch {epoch}: loss={logs['train_loss']:.4f}")
        ...
        >>> callback = PrintEpochCallback()
        >>> trainer = get_trainer(..., callbacks=[callback])
        >>> trainer.train()
    """

    def on_train_begin(self, trainer):
        """
        Called at the beginning of training.

        Args:
            trainer: The trainer instance (BaseTrainer or subclass)

        Example:
            >>> def on_train_begin(self, trainer):
            ...     print(f"Starting training for {trainer.num_epochs} epochs")
            ...     self.start_time = time.time()
        """
        pass

    def on_train_end(self, trainer):
        """
        Called at the end of training.

        Args:
            trainer: The trainer instance

        Example:
            >>> def on_train_end(self, trainer):
            ...     elapsed = time.time() - self.start_time
            ...     print(f"Training completed in {elapsed:.2f} seconds")
        """
        pass

    def on_epoch_begin(self, trainer, epoch):
        """
        Called at the beginning of each epoch.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number (0-indexed)

        Example:
            >>> def on_epoch_begin(self, trainer, epoch):
            ...     print(f"Starting epoch {epoch}/{trainer.num_epochs - 1}")
        """
        pass

    def on_epoch_end(self, trainer, epoch, logs):
        """
        Called at the end of each epoch after both train and val phases.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number (0-indexed)
            logs: Dictionary of metrics for this epoch, e.g.:
                {
                    'train_loss': 0.5,
                    'train_acc': 0.85,
                    'val_loss': 0.6,
                    'val_acc': 0.82,
                    'lr': 0.001
                }

        Example:
            >>> def on_epoch_end(self, trainer, epoch, logs):
            ...     if logs['val_acc'] > self.best_acc:
            ...         self.best_acc = logs['val_acc']
            ...         print(f"New best accuracy: {self.best_acc:.4f}")
        """
        pass

    def on_phase_begin(self, trainer, phase):
        """
        Called at the beginning of each phase (train or val).

        Args:
            trainer: The trainer instance
            phase: Phase name ('train' or 'val')

        Example:
            >>> def on_phase_begin(self, trainer, phase):
            ...     if phase == 'train':
            ...         self.train_batch_count = 0
        """
        pass

    def on_phase_end(self, trainer, phase, logs):
        """
        Called at the end of each phase (train or val).

        Args:
            trainer: The trainer instance
            phase: Phase name ('train' or 'val')
            logs: Dictionary of metrics for this phase, e.g.:
                {'loss': 0.5, 'acc': 0.85}

        Example:
            >>> def on_phase_end(self, trainer, phase, logs):
            ...     if phase == 'val':
            ...         print(f"Validation acc: {logs['acc']:.4f}")
        """
        pass

    def on_batch_begin(self, trainer, batch_idx, batch):
        """
        Called at the beginning of each batch.

        Args:
            trainer: The trainer instance
            batch_idx: Batch index within the epoch
            batch: The batch data (inputs, labels) tuple

        Example:
            >>> def on_batch_begin(self, trainer, batch_idx, batch):
            ...     inputs, labels = batch
            ...     # Apply custom augmentation to inputs
            ...     inputs = self.custom_augment(inputs)
        """
        pass

    def on_batch_end(self, trainer, batch_idx, batch, outputs, loss):
        """
        Called at the end of each batch after forward and backward passes.

        Args:
            trainer: The trainer instance
            batch_idx: Batch index within the epoch
            batch: The batch data (inputs, labels) tuple
            outputs: Model outputs (logits)
            loss: Computed loss value (tensor)

        Example:
            >>> def on_batch_end(self, trainer, batch_idx, batch, outputs, loss):
            ...     if batch_idx % 100 == 0:
            ...         print(f"Batch {batch_idx}: loss={loss.item():.4f}")
        """
        pass

    def on_backward_begin(self, trainer, loss):
        """
        Called before the backward pass.

        Args:
            trainer: The trainer instance
            loss: Loss tensor before backward()

        Example:
            >>> def on_backward_begin(self, trainer, loss):
            ...     # Log loss value before backward
            ...     self.loss_history.append(loss.item())
        """
        pass

    def on_backward_end(self, trainer):
        """
        Called after the backward pass but before optimizer step.

        Useful for gradient manipulation (clipping, logging, etc.).

        Args:
            trainer: The trainer instance

        Example:
            >>> def on_backward_end(self, trainer):
            ...     # Clip gradients
            ...     torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        """
        pass

    def on_optimizer_step_begin(self, trainer):
        """
        Called before the optimizer step.

        Args:
            trainer: The trainer instance

        Example:
            >>> def on_optimizer_step_begin(self, trainer):
            ...     # Log gradient statistics
            ...     total_norm = 0
            ...     for p in trainer.model.parameters():
            ...         if p.grad is not None:
            ...             total_norm += p.grad.data.norm(2).item() ** 2
            ...     print(f"Gradient norm: {total_norm ** 0.5:.4f}")
        """
        pass

    def on_optimizer_step_end(self, trainer):
        """
        Called after the optimizer step.

        Args:
            trainer: The trainer instance

        Example:
            >>> def on_optimizer_step_end(self, trainer):
            ...     # Update EMA model
            ...     self.ema_model.update(trainer.model)
        """
        pass


class CallbackManager:
    """
    Manager for orchestrating multiple callbacks.

    Provides a convenient interface for invoking hooks on multiple callbacks
    in the correct order. Callbacks are executed sequentially in the order they
    were registered.

    Attributes:
        callbacks: List of Callback instances

    Example:
        >>> callbacks = [EarlyStoppingCallback(), ModelCheckpointCallback()]
        >>> manager = CallbackManager(callbacks)
        >>> manager.on_train_begin(trainer)
        >>> manager.on_epoch_end(trainer, epoch, logs)
    """

    def __init__(self, callbacks=None):
        """
        Initialize the callback manager.

        Args:
            callbacks: List of Callback instances. Defaults to empty list.
        """
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer):
        """Call on_train_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer):
        """Call on_train_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, trainer, epoch):
        """Call on_epoch_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer, epoch, logs):
        """Call on_epoch_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)

    def on_phase_begin(self, trainer, phase):
        """Call on_phase_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_phase_begin(trainer, phase)

    def on_phase_end(self, trainer, phase, logs):
        """Call on_phase_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_phase_end(trainer, phase, logs)

    def on_batch_begin(self, trainer, batch_idx, batch):
        """Call on_batch_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx, batch)

    def on_batch_end(self, trainer, batch_idx, batch, outputs, loss):
        """Call on_batch_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, batch, outputs, loss)

    def on_backward_begin(self, trainer, loss):
        """Call on_backward_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_backward_begin(trainer, loss)

    def on_backward_end(self, trainer):
        """Call on_backward_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_backward_end(trainer)

    def on_optimizer_step_begin(self, trainer):
        """Call on_optimizer_step_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_optimizer_step_begin(trainer)

    def on_optimizer_step_end(self, trainer):
        """Call on_optimizer_step_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_optimizer_step_end(trainer)

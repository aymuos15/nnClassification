"""Progress bar callback for visualizing training progress."""

from tqdm import tqdm

from ml_src.core.callbacks.base import Callback


class ProgressBar(Callback):
    """
    Display training progress with tqdm progress bars.

    Creates progress bars for both epochs and batches, displaying current metrics
    and estimated time remaining.

    Attributes:
        show_metrics: Whether to display loss/accuracy in progress bar
        position: Progress bar position (useful for multi-process training)
        epoch_bar: tqdm progress bar for epochs
        batch_bar: tqdm progress bar for batches

    Example:
        >>> # In config.yaml:
        >>> # training:
        >>> #   callbacks:
        >>> #     - type: 'progress_bar'
        >>> #       show_metrics: true
        >>> #       position: 0
        >>>
        >>> from ml_src.core.callbacks import get_callbacks
        >>> callbacks = get_callbacks(config)
        >>> trainer = get_trainer(..., callbacks=callbacks)
        >>> trainer.train()  # Shows progress bars during training
    """

    def __init__(self, show_metrics=True, position=0):
        """
        Initialize progress bar callback.

        Args:
            show_metrics: Whether to display metrics in progress bar
            position: Progress bar position (0 for main process)
        """
        super().__init__()
        self.show_metrics = show_metrics
        self.position = position
        self.epoch_bar = None
        self.batch_bar = None
        self.current_phase = None

    def on_train_begin(self, trainer):
        """
        Create epoch-level progress bar.

        Args:
            trainer: The trainer instance
        """
        self.epoch_bar = tqdm(
            total=trainer.num_epochs,
            desc="Training",
            position=self.position,
            leave=True,
            ncols=100,
        )

    def on_epoch_begin(self, trainer, epoch):
        """
        Update epoch progress bar.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
        """
        if self.epoch_bar is not None:
            self.epoch_bar.set_description(f"Epoch {epoch}/{trainer.num_epochs - 1}")

    def on_phase_begin(self, trainer, phase):
        """
        Create batch-level progress bar for current phase.

        Args:
            trainer: The trainer instance
            phase: Phase name ('train' or 'val')
        """
        self.current_phase = phase

        # Close previous batch bar if it exists
        if self.batch_bar is not None:
            self.batch_bar.close()

        # Create new batch bar
        total_batches = len(trainer.dataloaders[phase])
        self.batch_bar = tqdm(
            total=total_batches,
            desc=f"{phase.capitalize()}",
            position=self.position + 1,
            leave=False,
            ncols=100,
        )

    def on_batch_end(self, trainer, batch_idx, batch, outputs, loss):
        """
        Update batch progress bar with current metrics.

        Args:
            trainer: The trainer instance
            batch_idx: Batch index
            batch: The batch data
            outputs: Model outputs
            loss: Computed loss
        """
        if self.batch_bar is not None:
            # Update progress
            self.batch_bar.update(1)

            # Update postfix with metrics if enabled
            if self.show_metrics:
                postfix = {"loss": f"{loss.item():.4f}"}
                self.batch_bar.set_postfix(postfix)

    def on_phase_end(self, trainer, phase, logs):
        """
        Close batch progress bar and display final metrics.

        Args:
            trainer: The trainer instance
            phase: Phase name
            logs: Dictionary of metrics
        """
        if self.batch_bar is not None:
            # Update with final metrics
            if self.show_metrics:
                postfix = {
                    "loss": f"{logs['loss']:.4f}",
                    "acc": f"{logs['acc']:.4f}",
                }
                self.batch_bar.set_postfix(postfix)

            self.batch_bar.close()
            self.batch_bar = None

    def on_epoch_end(self, trainer, epoch, logs):
        """
        Update epoch progress bar with epoch metrics.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            logs: Dictionary of epoch metrics
        """
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)

            # Update postfix with epoch metrics if enabled
            if self.show_metrics:
                postfix = {
                    "train_loss": f"{logs.get('train_loss', 0):.4f}",
                    "val_loss": f"{logs.get('val_loss', 0):.4f}",
                    "train_acc": f"{logs.get('train_acc', 0):.4f}",
                    "val_acc": f"{logs.get('val_acc', 0):.4f}",
                }
                self.epoch_bar.set_postfix(postfix)

    def on_train_end(self, trainer):
        """
        Close all progress bars.

        Args:
            trainer: The trainer instance
        """
        if self.batch_bar is not None:
            self.batch_bar.close()
            self.batch_bar = None

        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None

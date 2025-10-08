"""Early stopping callback to prevent overfitting during training."""

from loguru import logger

from ml_src.core.callbacks.base import Callback


class EarlyStoppingCallback(Callback):
    """
    Stop training when a monitored metric stops improving.

    Monitors a validation metric (accuracy or loss) and triggers early stopping if no improvement
    is observed for a specified number of epochs (patience). Sets trainer.should_stop flag when
    patience is exhausted.

    Attributes:
        monitor: Metric to monitor ('val_acc' or 'val_loss')
        patience: Number of epochs to wait for improvement before stopping
        mode: 'max' for metrics to maximize (accuracy), 'min' for metrics to minimize (loss)
        min_delta: Minimum change in monitored metric to qualify as improvement
        best_value: Best metric value observed so far
        counter: Number of epochs since last improvement
        stopped_epoch: Epoch at which training was stopped (None if not stopped)

    Example:
        >>> # In config.yaml:
        >>> # training:
        >>> #   callbacks:
        >>> #     - type: 'early_stopping'
        >>> #       monitor: 'val_acc'
        >>> #       patience: 10
        >>> #       mode: 'max'
        >>> #       min_delta: 0.001
        >>>
        >>> from ml_src.core.callbacks import get_callbacks
        >>> callbacks = get_callbacks(config)
        >>> trainer = get_trainer(..., callbacks=callbacks)
        >>> trainer.train()  # Will stop early if no improvement
    """

    def __init__(self, monitor="val_acc", patience=10, mode="max", min_delta=0.0):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor ('val_acc' or 'val_loss')
            patience: Number of epochs to wait for improvement before stopping
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            min_delta: Minimum change in monitored metric to qualify as improvement
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.stopped_epoch = None

        # Validate mode
        if mode not in ["max", "min"]:
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        # Validate monitor
        if monitor not in ["val_acc", "val_loss"]:
            logger.warning(
                f"Monitor metric '{monitor}' is non-standard. "
                "Recommended: 'val_acc' or 'val_loss'"
            )

        # Warn if mode doesn't match metric convention
        if monitor == "val_acc" and mode != "max":
            logger.warning(
                f"Metric '{monitor}' typically uses mode='max', but mode='{mode}' was specified"
            )
        if monitor == "val_loss" and mode != "min":
            logger.warning(
                f"Metric '{monitor}' typically uses mode='min', but mode='{mode}' was specified"
            )

    def _is_improvement(self, current_value):
        """
        Check if current value is an improvement over best value.

        Args:
            current_value: Current metric value to check

        Returns:
            bool: True if current value is an improvement
        """
        if self.best_value is None:
            return True

        if self.mode == "max":
            return current_value > self.best_value + self.min_delta
        else:  # mode == 'min'
            return current_value < self.best_value - self.min_delta

    def on_epoch_end(self, trainer, epoch, logs):
        """
        Check for improvement and trigger early stopping if necessary.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            logs: Dictionary of metrics including the monitored metric
        """
        # Get current value of monitored metric
        current_value = logs.get(self.monitor)

        if current_value is None:
            logger.warning(
                f"Early stopping metric '{self.monitor}' not found in logs. "
                f"Available: {list(logs.keys())}"
            )
            return

        # Check for improvement
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.counter = 0
            logger.debug(
                f"Early stopping: new best {self.monitor}={current_value:.4f}, "
                f"counter reset to 0"
            )
        else:
            self.counter += 1
            logger.debug(
                f"Early stopping: no improvement ({self.counter}/{self.patience}), "
                f"best {self.monitor}={self.best_value:.4f}"
            )

            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                logger.opt(colors=True).warning(
                    f"<yellow>Early stopping triggered at epoch {epoch}</yellow> "
                    f"({self.monitor} did not improve for {self.patience} epochs)"
                )
                # Set flag to stop training
                trainer.should_stop = True

    def get_state(self):
        """
        Get current state for checkpointing.

        Returns:
            dict: State dictionary containing all attributes
        """
        return {
            "monitor": self.monitor,
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "best_value": self.best_value,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state(self, state):
        """
        Load state from checkpoint.

        Args:
            state: State dictionary from checkpoint
        """
        self.monitor = state["monitor"]
        self.patience = state["patience"]
        self.mode = state["mode"]
        self.min_delta = state["min_delta"]
        self.counter = state["counter"]
        self.best_value = state["best_value"]
        self.stopped_epoch = state["stopped_epoch"]

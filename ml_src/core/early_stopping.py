"""Early stopping utility for preventing overfitting during training."""

from loguru import logger


class EarlyStopping:
    """
    Early stopping handler to stop training when validation metric stops improving.

    Monitors a validation metric (accuracy or loss) and stops training if no improvement
    is observed for a specified number of epochs (patience). Supports both maximization
    (for accuracy) and minimization (for loss) modes.

    Attributes:
        patience: Number of epochs to wait for improvement before stopping
        metric: Which metric to monitor ('val_acc' or 'val_loss')
        mode: 'max' for metrics to maximize (accuracy), 'min' for metrics to minimize (loss)
        min_delta: Minimum change in monitored metric to qualify as improvement
        best_value: Best metric value observed so far
        counter: Number of epochs since last improvement
        stopped_epoch: Epoch at which training was stopped (None if not stopped)

    Example:
        >>> early_stopping = EarlyStopping(patience=10, metric='val_acc', mode='max')
        >>> for epoch in range(num_epochs):
        ...     val_acc = train_one_epoch()
        ...     if early_stopping.should_stop(epoch, val_acc):
        ...         print(f"Early stopping triggered at epoch {epoch}")
        ...         break
    """

    def __init__(self, patience=10, metric="val_acc", mode="max", min_delta=0.0):
        """
        Initialize early stopping handler.

        Args:
            patience: Number of epochs to wait for improvement before stopping
            metric: Which metric to monitor ('val_acc' or 'val_loss')
            mode: 'max' for metrics to maximize (accuracy), 'min' for metrics to minimize (loss)
            min_delta: Minimum change in monitored metric to qualify as improvement
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.stopped_epoch = None

        # Validate mode
        if mode not in ["max", "min"]:
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        # Validate metric
        if metric not in ["val_acc", "val_loss"]:
            raise ValueError(f"metric must be 'val_acc' or 'val_loss', got '{metric}'")

        # Ensure mode matches metric convention
        if metric == "val_acc" and mode != "max":
            logger.warning(
                f"Metric '{metric}' typically uses mode='max', but mode='{mode}' was specified"
            )
        if metric == "val_loss" and mode != "min":
            logger.warning(
                f"Metric '{metric}' typically uses mode='min', but mode='{mode}' was specified"
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

    def should_stop(self, epoch, current_value):
        """
        Check if training should stop based on current metric value.

        Args:
            epoch: Current epoch number
            current_value: Current value of the monitored metric

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                logger.opt(colors=True).warning(
                    f"<yellow>Early stopping triggered at epoch {epoch}</yellow> "
                    f"({self.metric} did not improve for {self.patience} epochs)"
                )
                return True

        return False

    def get_state(self):
        """
        Get current state for checkpointing.

        Returns:
            dict: State dictionary containing all attributes
        """
        return {
            "patience": self.patience,
            "metric": self.metric,
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
        self.patience = state["patience"]
        self.metric = state["metric"]
        self.mode = state["mode"]
        self.min_delta = state["min_delta"]
        self.counter = state["counter"]
        self.best_value = state["best_value"]
        self.stopped_epoch = state["stopped_epoch"]

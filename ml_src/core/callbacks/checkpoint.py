"""Model checkpointing callback for saving best models during training."""

import os
from pathlib import Path

from loguru import logger

from ml_src.core.callbacks.base import Callback


class ModelCheckpointCallback(Callback):
    """
    Save model checkpoints based on monitored metric.

    Monitors a metric (e.g., val_acc) and saves the top-k best models to disk.
    Automatically manages checkpoint files, keeping only the best k models and
    optionally the last checkpoint.

    Attributes:
        monitor: Metric to monitor for determining best models
        mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        save_top_k: Number of best models to keep (default: 1)
        save_last: Whether to always save the last checkpoint (default: True)
        dirpath: Directory to save checkpoints (defaults to trainer.run_dir/weights)
        filename: Filename pattern for checkpoints (supports {epoch}, {monitor}, etc.)
        saved_checkpoints: List of (metric_value, filepath) tuples for saved models

    Example:
        >>> # In config.yaml:
        >>> # training:
        >>> #   callbacks:
        >>> #     - type: 'model_checkpoint'
        >>> #       monitor: 'val_acc'
        >>> #       mode: 'max'
        >>> #       save_top_k: 3
        >>> #       filename: 'epoch_{epoch:02d}_acc_{val_acc:.4f}.pt'
        >>>
        >>> from ml_src.core.callbacks import get_callbacks
        >>> callbacks = get_callbacks(config)
        >>> trainer = get_trainer(..., callbacks=callbacks)
        >>> trainer.train()  # Automatically saves top-3 models
    """

    def __init__(
        self,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=None,
        filename="best.pt",
    ):
        """
        Initialize model checkpoint callback.

        Args:
            monitor: Metric to monitor for determining best models
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            save_top_k: Number of best models to keep (1 = only best, -1 = save all)
            save_last: Whether to always save the last checkpoint
            dirpath: Directory to save checkpoints (None = use trainer.run_dir/weights)
            filename: Filename pattern (supports {epoch}, {monitor_name}, metric values)
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.dirpath = dirpath
        self.filename = filename
        self.saved_checkpoints = []  # List of (metric_value, filepath)

        # Validate mode
        if mode not in ["max", "min"]:
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        # Warn about common misconfigurations
        if monitor == "val_acc" and mode != "max":
            logger.warning(
                f"Metric '{monitor}' typically uses mode='max', but mode='{mode}' was specified"
            )
        if monitor == "val_loss" and mode != "min":
            logger.warning(
                f"Metric '{monitor}' typically uses mode='min', but mode='{mode}' was specified"
            )

    def _is_better(self, current, best):
        """Check if current value is better than best value."""
        if best is None:
            return True
        if self.mode == "max":
            return current > best
        else:
            return current < best

    def _format_filename(self, epoch, logs):
        """Format filename with epoch and metric values."""
        # Create format dict with epoch and all metrics
        format_dict = {"epoch": epoch}
        format_dict.update(logs)

        try:
            return self.filename.format(**format_dict)
        except KeyError as e:
            logger.warning(f"Filename formatting failed: {e}. Using default format.")
            return f"epoch_{epoch:02d}.pt"

    def _should_save(self, metric_value):
        """Determine if current model should be saved based on metric value."""
        if self.save_top_k == -1:
            # Save all checkpoints
            return True

        if len(self.saved_checkpoints) < self.save_top_k:
            # Haven't saved enough checkpoints yet
            return True

        # Check if current is better than worst saved checkpoint
        saved_values = [val for val, _ in self.saved_checkpoints]
        if self.mode == "max":
            worst_saved = min(saved_values)
            return metric_value > worst_saved
        else:
            worst_saved = max(saved_values)
            return metric_value < worst_saved

    def _cleanup_checkpoints(self):
        """Remove worst checkpoint to maintain save_top_k limit."""
        if self.save_top_k == -1 or len(self.saved_checkpoints) <= self.save_top_k:
            return

        # Find worst checkpoint
        if self.mode == "max":
            worst_idx = min(range(len(self.saved_checkpoints)),
                          key=lambda i: self.saved_checkpoints[i][0])
        else:
            worst_idx = max(range(len(self.saved_checkpoints)),
                          key=lambda i: self.saved_checkpoints[i][0])

        # Remove worst checkpoint file
        _, filepath = self.saved_checkpoints[worst_idx]
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Removed old checkpoint: {filepath}")

        # Remove from list
        del self.saved_checkpoints[worst_idx]

    def on_epoch_end(self, trainer, epoch, logs):
        """
        Save checkpoint if current model is among top-k best.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            logs: Dictionary of metrics including the monitored metric
        """
        # Get current value of monitored metric
        metric_value = logs.get(self.monitor)

        if metric_value is None:
            logger.warning(
                f"Checkpoint metric '{self.monitor}' not found in logs. "
                f"Available: {list(logs.keys())}"
            )
            return

        # Determine checkpoint directory
        if self.dirpath is None:
            dirpath = os.path.join(trainer.run_dir, "weights")
        else:
            dirpath = self.dirpath

        # Create directory if it doesn't exist
        Path(dirpath).mkdir(parents=True, exist_ok=True)

        # Check if we should save this checkpoint
        if self._should_save(metric_value):
            # Format filename
            filename = self._format_filename(epoch, logs)
            filepath = os.path.join(dirpath, filename)

            # Save checkpoint
            metrics = {
                "train_losses": trainer.train_losses if hasattr(trainer, 'train_losses') else [],
                "val_losses": trainer.val_losses if hasattr(trainer, 'val_losses') else [],
                "train_accs": trainer.train_accs if hasattr(trainer, 'train_accs') else [],
                "val_accs": trainer.val_accs if hasattr(trainer, 'val_accs') else [],
            }

            # Use trainer's save_checkpoint method
            best_acc = logs.get("val_acc", 0.0)
            trainer.save_checkpoint(epoch, best_acc, metrics, filepath)

            # Track saved checkpoint
            self.saved_checkpoints.append((metric_value, filepath))
            logger.success(
                f"Saved checkpoint: {filename} ({self.monitor}={metric_value:.4f})"
            )

            # Cleanup old checkpoints
            self._cleanup_checkpoints()

        # Save last checkpoint if enabled
        if self.save_last:
            last_path = os.path.join(dirpath, "last.pt")
            metrics = {
                "train_losses": trainer.train_losses if hasattr(trainer, 'train_losses') else [],
                "val_losses": trainer.val_losses if hasattr(trainer, 'val_losses') else [],
                "train_accs": trainer.train_accs if hasattr(trainer, 'train_accs') else [],
                "val_accs": trainer.val_accs if hasattr(trainer, 'val_accs') else [],
            }
            best_acc = logs.get("val_acc", 0.0)
            trainer.save_checkpoint(epoch, best_acc, metrics, last_path)

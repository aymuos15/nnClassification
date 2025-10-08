"""Learning rate monitoring callback for tracking LR during training."""

from loguru import logger

from ml_src.core.callbacks.base import Callback


class LearningRateMonitor(Callback):
    """
    Monitor and log learning rate to TensorBoard.

    Tracks learning rate(s) from the optimizer and logs them to TensorBoard at each epoch.
    Supports multiple parameter groups for discriminative learning rates.

    Attributes:
        log_momentum: Whether to also log momentum values (default: False)
        log_to_console: Whether to print LR to console (default: False)

    Example:
        >>> # In config.yaml:
        >>> # training:
        >>> #   callbacks:
        >>> #     - type: 'lr_monitor'
        >>> #       log_momentum: true
        >>> #       log_to_console: false
        >>>
        >>> from ml_src.core.callbacks import get_callbacks
        >>> callbacks = get_callbacks(config)
        >>> trainer = get_trainer(..., callbacks=callbacks)
        >>> trainer.train()  # LR automatically logged to TensorBoard
    """

    def __init__(self, log_momentum=False, log_to_console=False):
        """
        Initialize learning rate monitor callback.

        Args:
            log_momentum: Whether to log momentum values
            log_to_console: Whether to print LR to console
        """
        super().__init__()
        self.log_momentum = log_momentum
        self.log_to_console = log_to_console

    def on_epoch_end(self, trainer, epoch, logs):
        """
        Log learning rate(s) to TensorBoard.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        # Log learning rate for each parameter group
        for idx, param_group in enumerate(trainer.optimizer.param_groups):
            lr = param_group.get("lr")

            if lr is not None:
                # Determine tag name
                if len(trainer.optimizer.param_groups) == 1:
                    tag = "Learning_Rate"
                else:
                    tag = f"Learning_Rate/group_{idx}"

                # Log to TensorBoard
                trainer.writer.add_scalar(tag, lr, epoch)

                # Optionally log to console
                if self.log_to_console:
                    if len(trainer.optimizer.param_groups) == 1:
                        logger.info(f"LR: {lr:.6f}")
                    else:
                        logger.info(f"LR group {idx}: {lr:.6f}")

            # Log momentum if enabled
            if self.log_momentum:
                momentum = param_group.get("momentum")
                if momentum is not None:
                    if len(trainer.optimizer.param_groups) == 1:
                        tag = "Momentum"
                    else:
                        tag = f"Momentum/group_{idx}"

                    trainer.writer.add_scalar(tag, momentum, epoch)

                    if self.log_to_console:
                        if len(trainer.optimizer.param_groups) == 1:
                            logger.info(f"Momentum: {momentum:.4f}")
                        else:
                            logger.info(f"Momentum group {idx}: {momentum:.4f}")

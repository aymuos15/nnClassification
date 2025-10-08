"""Gradient-related callbacks for clipping and monitoring."""

import torch
from loguru import logger

from ml_src.core.callbacks.base import Callback


class GradientClipping(Callback):
    """
    Clip gradients to prevent exploding gradients.

    Applies gradient clipping after the backward pass but before the optimizer step.
    Supports both norm-based and value-based clipping.

    Attributes:
        value: Clipping threshold
        algorithm: Clipping algorithm ('norm' or 'value')
        norm_type: Type of norm for norm-based clipping (default: 2.0 for L2 norm)

    Example:
        >>> # In config.yaml:
        >>> # training:
        >>> #   callbacks:
        >>> #     - type: 'gradient_clipping'
        >>> #       value: 1.0
        >>> #       algorithm: 'norm'
        >>> #       norm_type: 2.0
        >>>
        >>> from ml_src.core.callbacks import get_callbacks
        >>> callbacks = get_callbacks(config)
        >>> trainer = get_trainer(..., callbacks=callbacks)
        >>> trainer.train()  # Gradients automatically clipped
    """

    def __init__(self, value=1.0, algorithm="norm", norm_type=2.0):
        """
        Initialize gradient clipping callback.

        Args:
            value: Clipping threshold
            algorithm: 'norm' for gradient norm clipping, 'value' for gradient value clipping
            norm_type: Type of norm (2.0 for L2, float('inf') for max norm)
        """
        super().__init__()
        self.value = value
        self.algorithm = algorithm
        self.norm_type = norm_type

        # Validate algorithm
        if algorithm not in ["norm", "value"]:
            raise ValueError(f"algorithm must be 'norm' or 'value', got '{algorithm}'")

    def on_backward_end(self, trainer):
        """
        Clip gradients after backward pass.

        Args:
            trainer: The trainer instance
        """
        if self.algorithm == "norm":
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(),
                max_norm=self.value,
                norm_type=self.norm_type,
            )
        elif self.algorithm == "value":
            # Clip gradient values
            torch.nn.utils.clip_grad_value_(
                trainer.model.parameters(),
                clip_value=self.value,
            )


class GradientNormMonitor(Callback):
    """
    Monitor and log gradient norms to TensorBoard.

    Computes gradient norms and logs them to TensorBoard for monitoring gradient flow.
    Useful for detecting vanishing or exploding gradients.

    Attributes:
        frequency: Log frequency in batches (default: 10)
        norm_type: Type of norm to compute (default: 2.0 for L2 norm)
        log_per_layer: Whether to log per-layer gradients (default: False)
        batch_counter: Counter for batch frequency

    Example:
        >>> # In config.yaml:
        >>> # training:
        >>> #   callbacks:
        >>> #     - type: 'gradient_norm_monitor'
        >>> #       frequency: 10
        >>> #       norm_type: 2.0
        >>> #       log_per_layer: false
        >>>
        >>> from ml_src.core.callbacks import get_callbacks
        >>> callbacks = get_callbacks(config)
        >>> trainer = get_trainer(..., callbacks=callbacks)
        >>> trainer.train()  # Gradient norms logged to TensorBoard
    """

    def __init__(self, frequency=10, norm_type=2.0, log_per_layer=False):
        """
        Initialize gradient norm monitor callback.

        Args:
            frequency: Log frequency in batches
            norm_type: Type of norm to compute (2.0 for L2, float('inf') for max)
            log_per_layer: Whether to log gradients for each layer separately
        """
        super().__init__()
        self.frequency = frequency
        self.norm_type = norm_type
        self.log_per_layer = log_per_layer
        self.batch_counter = 0
        self.epoch = 0

    def on_epoch_begin(self, trainer, epoch):
        """
        Reset batch counter at start of epoch.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
        """
        self.batch_counter = 0
        self.epoch = epoch

    def on_backward_end(self, trainer):
        """
        Compute and log gradient norms.

        Args:
            trainer: The trainer instance
        """
        self.batch_counter += 1

        # Only log at specified frequency
        if self.batch_counter % self.frequency != 0:
            return

        # Compute total gradient norm
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.norm_type).item()
                total_norm += param_norm ** self.norm_type

        total_norm = total_norm ** (1.0 / self.norm_type)

        # Log to TensorBoard
        global_step = self.epoch * len(trainer.dataloaders["train"]) + self.batch_counter
        trainer.writer.add_scalar("Gradients/total_norm", total_norm, global_step)

        # Optionally log per-layer gradients
        if self.log_per_layer:
            for name, param in trainer.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(self.norm_type).item()
                    # Clean up parameter name for TensorBoard
                    clean_name = name.replace(".", "/")
                    trainer.writer.add_scalar(
                        f"Gradients/layer/{clean_name}",
                        grad_norm,
                        global_step,
                    )

        # Warn if gradients are vanishing or exploding
        if total_norm < 1e-7:
            logger.warning(
                f"Very small gradient norm detected: {total_norm:.2e}. "
                "Possible vanishing gradients."
            )
        elif total_norm > 100:
            logger.warning(
                f"Very large gradient norm detected: {total_norm:.2e}. "
                "Possible exploding gradients. Consider gradient clipping."
            )

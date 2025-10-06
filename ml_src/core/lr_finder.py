"""Learning Rate Finder utilities and visualization."""

import copy
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger


class LRFinder:
    """
    Learning Rate Finder using the fastai LR range test methodology.

    This class implements the LR range test to find an optimal learning rate for training.
    It performs a mock training run where the learning rate increases exponentially from
    a very small value to a large value, tracking the loss at each step. The optimal LR
    is typically found at the steepest descent in the loss curve.

    Example:
        >>> finder = LRFinder()
        >>> lrs, losses, suggested_lr = finder.find_lr(
        ...     model, train_loader, optimizer_fn, criterion, device
        ... )
        >>> logger.info(f"Suggested LR: {suggested_lr}")
    """

    def find_lr(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer_fn: Callable,
        criterion: torch.nn.Module,
        device: torch.device,
        start_lr: float = 1e-8,
        end_lr: float = 10,
        num_iter: int = 100,
        beta: float = 0.98,
        diverge_threshold: float = 4.0,
    ) -> Tuple[list, list, float]:
        """
        Find optimal learning rate using the LR range test.

        Args:
            model: PyTorch model to test
            train_loader: DataLoader for training data
            optimizer_fn: Function that creates optimizer given learning rate
                         Example: lambda lr: torch.optim.SGD(model.parameters(), lr=lr)
            criterion: Loss function
            device: Device to run on (cpu or cuda)
            start_lr: Starting learning rate (default: 1e-8)
            end_lr: Ending learning rate (default: 10)
            num_iter: Number of iterations to run (default: 100)
            beta: Smoothing factor for exponential moving average (default: 0.98)
            diverge_threshold: Multiplier for early stopping when loss > threshold * min_loss (default: 4.0)

        Returns:
            Tuple of (lr_list, loss_list, suggested_lr):
                - lr_list: List of learning rates tested
                - loss_list: List of smoothed losses
                - suggested_lr: Suggested optimal learning rate

        Example:
            >>> optimizer_fn = lambda lr: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            >>> lrs, losses, suggested_lr = finder.find_lr(
            ...     model, train_loader, optimizer_fn, criterion, device,
            ...     start_lr=1e-7, end_lr=1, num_iter=100
            ... )
        """
        # Validate inputs
        if len(train_loader) == 0:
            logger.error("Train loader is empty")
            raise ValueError("Train loader must contain at least one batch")

        if start_lr >= end_lr:
            logger.error(f"start_lr ({start_lr}) must be less than end_lr ({end_lr})")
            raise ValueError("start_lr must be less than end_lr")

        if num_iter <= 0:
            logger.error(f"num_iter must be positive, got {num_iter}")
            raise ValueError("num_iter must be positive")

        logger.info(
            f"Starting LR range test: {start_lr:.2e} -> {end_lr:.2e} over {num_iter} iterations"
        )

        # Save initial model state
        initial_state = copy.deepcopy(model.state_dict())
        logger.debug("Saved initial model state")

        # Create optimizer with start_lr
        optimizer = optimizer_fn(start_lr)

        # Set model to training mode
        model.train()

        # Initialize tracking variables
        lr_list = []
        loss_list = []
        smoothed_loss = 0
        best_loss = float("inf")
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        # Create iterator from train_loader
        train_iter = iter(train_loader)

        for iteration in range(num_iter):
            # Calculate current learning rate (exponential increase)
            lr = start_lr * (lr_mult**iteration)

            # Update optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            try:
                # Get batch from train_loader
                inputs, labels = next(train_iter)
            except StopIteration:
                # Restart iterator if we run out of data
                train_iter = iter(train_loader)
                inputs, labels = next(train_iter)

            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            try:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            except RuntimeError as e:
                logger.error(f"Forward pass failed at iteration {iteration}: {e}")
                # Restore model state and return empty results
                model.load_state_dict(initial_state)
                return [], [], start_lr

            # Check for NaN or Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at iteration {iteration}, LR={lr:.2e}")
                # Stop and restore model
                model.load_state_dict(initial_state)
                if len(lr_list) > 0:
                    # Return results up to this point
                    suggested_lr = self._find_optimal_lr(lr_list, loss_list)
                    return lr_list, loss_list, suggested_lr
                else:
                    return [], [], start_lr

            # Backward pass
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                logger.error(f"Backward pass failed at iteration {iteration}: {e}")
                model.load_state_dict(initial_state)
                return [], [], start_lr

            # Compute smoothed loss using exponential moving average
            loss_val = loss.item()
            if iteration == 0:
                smoothed_loss = loss_val
            else:
                smoothed_loss = beta * smoothed_loss + (1 - beta) * loss_val

            # Track minimum loss for divergence check
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Store learning rate and smoothed loss
            lr_list.append(lr)
            loss_list.append(smoothed_loss)

            # Early stopping: check for divergence (loss > diverge_threshold * min_loss)
            if smoothed_loss > diverge_threshold * best_loss:
                logger.warning(
                    f"Loss diverged at iteration {iteration}, LR={lr:.2e} "
                    f"(smoothed_loss={smoothed_loss:.4f}, best_loss={best_loss:.4f}, "
                    f"threshold={diverge_threshold}x)"
                )
                break

            # Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                logger.debug(
                    f"Iteration {iteration + 1}/{num_iter}, LR={lr:.2e}, Loss={smoothed_loss:.4f}"
                )

        # Restore initial model state
        model.load_state_dict(initial_state)
        logger.debug("Restored initial model state")

        # Find optimal learning rate
        if len(lr_list) == 0:
            logger.warning("No valid iterations completed, returning start_lr")
            return [], [], start_lr

        suggested_lr = self._find_optimal_lr(lr_list, loss_list)
        logger.success(f"LR range test complete. Suggested LR: {suggested_lr:.2e}")

        return lr_list, loss_list, suggested_lr

    def _find_optimal_lr(self, lr_list: list, loss_list: list) -> float:
        """
        Find optimal learning rate from the LR-loss curve.

        The optimal LR is found at the point of steepest descent in the loss curve,
        divided by 10 for safety (following fastai methodology).

        Args:
            lr_list: List of learning rates
            loss_list: List of smoothed losses

        Returns:
            Suggested optimal learning rate
        """
        if len(lr_list) < 3:
            # Not enough data points, return middle value
            logger.warning("Not enough data points for gradient calculation, returning middle LR")
            return lr_list[len(lr_list) // 2] if lr_list else 1e-3

        # Convert to numpy arrays for easier computation
        lrs = np.array(lr_list)
        losses = np.array(loss_list)

        # Compute gradient (derivative) of loss with respect to log(lr)
        # Use central difference for better accuracy
        gradients = np.gradient(losses, np.log10(lrs))

        # Find the learning rate with the steepest negative gradient
        # (most negative value = steepest descent)
        min_gradient_idx = np.argmin(gradients)
        optimal_lr = lrs[min_gradient_idx]

        # Apply safety factor of 10 (divide by 10)
        # This is the fastai recommendation
        suggested_lr = optimal_lr / 10

        logger.debug(
            f"Steepest descent at LR={optimal_lr:.2e}, suggested LR (÷10): {suggested_lr:.2e}"
        )

        return suggested_lr


def plot_lr_finder(lrs, losses, suggested_lr, save_path, title="Learning Rate Finder"):
    """
    Create and save a publication-quality learning rate finder plot.

    This function visualizes the relationship between learning rates and loss values,
    highlighting the suggested optimal learning rate for training.

    Args:
        lrs: List or array of learning rates (float)
        losses: List or array of corresponding loss values (float)
        suggested_lr: Suggested optimal learning rate (float)
        save_path: Path to save the plot (e.g., 'lr_finder.png')
        title: Plot title (default: "Learning Rate Finder")

    Returns:
        matplotlib.figure.Figure: Figure object containing the LR finder plot

    Example:
        >>> lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        >>> losses = [2.5, 2.3, 2.0, 1.5, 1.2, 2.0, 5.0]
        >>> suggested_lr = 1e-3
        >>> fig = plot_lr_finder(lrs, losses, suggested_lr, 'lr_finder.png')
    """
    # Convert to numpy arrays for easier manipulation
    lrs = np.array(lrs)
    losses = np.array(losses)

    # Create figure with high DPI for publication quality
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot learning rate vs loss
    ax.plot(lrs, losses, linewidth=2, color="#2E86AB", label="Loss")

    # Add vertical line at suggested learning rate
    ax.axvline(
        suggested_lr,
        color="#A23B72",
        linestyle="--",
        linewidth=2,
        label=f"Suggested LR: {suggested_lr:.2e}",
    )

    # Add annotation for suggested LR
    # Find the loss value at the suggested LR (or nearest)
    idx = np.abs(lrs - suggested_lr).argmin()
    loss_at_suggested = losses[idx]

    ax.annotate(
        f"LR = {suggested_lr:.2e}",
        xy=(suggested_lr, loss_at_suggested),
        xytext=(20, 20),
        textcoords="offset points",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFE5B4", "edgecolor": "#A23B72", "alpha": 0.8},
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.3", "color": "#A23B72"},
    )

    # Set log scale for x-axis
    ax.set_xscale("log")

    # Add grid for better readability
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Labels and title
    ax.set_xlabel("Learning Rate (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add legend
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Improve layout
    plt.tight_layout()

    # Save with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.success(f"Saved learning rate finder plot to {save_path}")

    return fig


def create_lr_finder_figure(lrs, losses, suggested_lr, title="Learning Rate Finder"):
    """
    Create a learning rate finder plot as a matplotlib figure for TensorBoard.

    Args:
        lrs: List or array of learning rates (float)
        losses: List or array of corresponding loss values (float)
        suggested_lr: Suggested optimal learning rate (float)
        title: Plot title (default: "Learning Rate Finder")

    Returns:
        matplotlib.figure.Figure: Figure object containing the LR finder plot

    Example:
        >>> lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        >>> losses = [2.5, 2.3, 2.0, 1.5, 1.2, 2.0, 5.0]
        >>> suggested_lr = 1e-3
        >>> fig = create_lr_finder_figure(lrs, losses, suggested_lr)
        >>> writer.add_figure('lr_finder', fig, 0)
        >>> plt.close(fig)
    """
    # Convert to numpy arrays for easier manipulation
    lrs = np.array(lrs)
    losses = np.array(losses)

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot learning rate vs loss
    ax.plot(lrs, losses, linewidth=2, color="#2E86AB", label="Loss")

    # Add vertical line at suggested learning rate
    ax.axvline(
        suggested_lr,
        color="#A23B72",
        linestyle="--",
        linewidth=2,
        label=f"Suggested LR: {suggested_lr:.2e}",
    )

    # Add annotation for suggested LR
    idx = np.abs(lrs - suggested_lr).argmin()
    loss_at_suggested = losses[idx]

    ax.annotate(
        f"LR = {suggested_lr:.2e}",
        xy=(suggested_lr, loss_at_suggested),
        xytext=(20, 20),
        textcoords="offset points",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFE5B4", "edgecolor": "#A23B72", "alpha": 0.8},
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.3", "color": "#A23B72"},
    )

    # Set log scale for x-axis
    ax.set_xscale("log")

    # Add grid
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Labels and title
    ax.set_xlabel("Learning Rate (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add legend
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Improve layout
    plt.tight_layout()

    return fig

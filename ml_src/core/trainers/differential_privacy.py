"""Differential Privacy trainer implementation using Opacus."""

import torch
from loguru import logger

from ml_src.core.checkpointing import load_checkpoint, save_checkpoint
from ml_src.core.trainers.base import BaseTrainer

# Handle opacus import gracefully
try:
    from opacus import PrivacyEngine

    _OPACUS_AVAILABLE = True
except ImportError:
    _OPACUS_AVAILABLE = False
    PrivacyEngine = None


class DPTrainer(BaseTrainer):
    """
    Differential Privacy trainer using Opacus.

    This trainer implements differentially private training with:
    - Privacy Engine for DP-SGD with noise injection
    - Gradient clipping for privacy guarantees
    - Privacy budget (epsilon) tracking
    - Automatic privacy accounting

    Requires: pip install opacus

    Privacy-Utility Tradeoff:
    - Higher noise_multiplier = More privacy, lower utility
    - Lower max_grad_norm = More privacy, slower convergence
    - Lower target_epsilon = Stronger privacy guarantees

    Example:
        >>> config = {
        ...     "training": {
        ...         "trainer_type": "dp",
        ...         "num_epochs": 10,
        ...         "batch_size": 32,
        ...         "dp": {
        ...             "noise_multiplier": 1.1,
        ...             "max_grad_norm": 1.0,
        ...             "target_epsilon": 3.0,
        ...             "target_delta": 1e-5,
        ...         }
        ...     }
        ... }
        >>> trainer = DPTrainer(
        ...     model=model,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     dataloaders=dataloaders,
        ...     dataset_sizes=dataset_sizes,
        ...     device=device,
        ...     config=config,
        ...     run_dir=run_dir,
        ...     class_names=class_names
        ... )
        >>> model, train_losses, val_losses, train_accs, val_accs = trainer.train()
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the DPTrainer.

        Raises:
            ImportError: If opacus is not installed
        """
        if not _OPACUS_AVAILABLE:
            raise ImportError(
                "DPTrainer requires opacus. Install with:\n"
                "  pip install opacus\n"
                "Or install with dp extras:\n"
                "  pip install -e '.[dp]'"
            )

        super().__init__(*args, **kwargs)

        # Extract DP configuration
        self.dp_config = self.config["training"].get("dp", {})
        self.noise_multiplier = self.dp_config.get("noise_multiplier", 1.1)
        self.max_grad_norm = self.dp_config.get("max_grad_norm", 1.0)
        self.target_epsilon = self.dp_config.get("target_epsilon", 3.0)
        self.target_delta = self.dp_config.get("target_delta", 1e-5)

        # PrivacyEngine will be initialized in prepare_training()
        self.privacy_engine = None

        logger.info("Initialized DPTrainer with Opacus")
        logger.info(f"  Noise Multiplier: {self.noise_multiplier}")
        logger.info(f"  Max Grad Norm: {self.max_grad_norm}")
        logger.info(f"  Target Epsilon: {self.target_epsilon}")
        logger.info(f"  Target Delta: {self.target_delta}")

    def prepare_training(self):
        """
        Prepare for differential privacy training.

        Initializes the PrivacyEngine and makes the model, optimizer, and
        dataloader privacy-aware using Opacus.
        """
        logger.info("Preparing PrivacyEngine for differential privacy training...")

        # Initialize PrivacyEngine
        self.privacy_engine = PrivacyEngine()

        # Make model, optimizer, and dataloader private
        self.model, self.optimizer, self.dataloaders["train"] = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloaders["train"],
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        logger.success("PrivacyEngine initialized successfully")
        logger.info(
            f"Training with DP-SGD (noise={self.noise_multiplier}, clip={self.max_grad_norm})"
        )

    def training_step(self, inputs, labels):
        """
        Execute a single training step with differential privacy.

        The PrivacyEngine automatically handles:
        - Per-sample gradient computation
        - Gradient clipping
        - Noise injection

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

        # Backward pass (PrivacyEngine hooks into this automatically)
        loss.backward()

        # Optimizer step (PrivacyEngine handles clipping and noise injection)
        self.optimizer.step()

        return outputs, loss

    def validation_step(self, inputs, labels):
        """
        Execute a single validation step.

        Validation is performed without privacy mechanisms (no noise/clipping).

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

    def _track_privacy_budget(self, epoch):
        """
        Compute and log privacy budget after an epoch.

        Args:
            epoch: Current epoch number
        """
        if self.privacy_engine is None:
            return

        # Compute epsilon with the target delta
        try:
            epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
        except (ValueError, RuntimeError):
            # No training steps yet, skip privacy tracking
            logger.debug("Skipping privacy budget tracking (no training steps yet)")
            return

        # Log to console
        logger.info(f"Privacy Budget (ε): {epsilon:.2f} at δ={self.target_delta}")

        # Log to TensorBoard
        self.writer.add_scalar("Privacy/epsilon", epsilon, epoch)
        self.writer.add_scalar("Privacy/delta", self.target_delta, epoch)

        # Warn if target epsilon exceeded
        if epsilon > self.target_epsilon:
            logger.warning(
                f"Privacy budget exceeded! Current ε={epsilon:.2f} > Target ε={self.target_epsilon}"
            )

    def save_checkpoint(self, epoch, best_acc, metrics, path):
        """
        Save a checkpoint including PrivacyEngine state.

        Extends standard checkpointing to include:
        - PrivacyEngine state (for privacy accounting)
        - DP configuration parameters
        - Early stopping state (if enabled)

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

        # First save the standard checkpoint
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
        )

        # Load the checkpoint and add PrivacyEngine state
        checkpoint = torch.load(path, weights_only=False)

        # Add PrivacyEngine accountant state if available
        if self.privacy_engine is not None:
            # Compute current epsilon for tracking (only if training has occurred)
            # get_epsilon can fail if no training steps have been taken yet
            try:
                epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
            except (ValueError, RuntimeError):
                # No training steps yet, epsilon is 0
                epsilon = 0.0

            checkpoint["privacy_engine_state"] = {
                "accountant": self.privacy_engine.accountant.state_dict()
                if hasattr(self.privacy_engine, "accountant")
                else None,
                "epsilon": epsilon,
                "delta": self.target_delta,
                "noise_multiplier": self.noise_multiplier,
                "max_grad_norm": self.max_grad_norm,
            }

        # Save updated checkpoint
        torch.save(checkpoint, path)
        logger.debug(f"Saved DP checkpoint with privacy state to {path}")

    def load_checkpoint(self, path):
        """
        Load a checkpoint including PrivacyEngine state.

        Restores both standard training state and PrivacyEngine accounting.

        Args:
            path: Path to the checkpoint file

        Returns:
            Tuple of (epoch, best_acc, train_losses, val_losses, train_accs, val_accs)
        """
        # Load standard checkpoint
        (
            epoch,
            best_acc,
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            config,
            early_stopping_state,
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

        # Load PrivacyEngine state if available
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if "privacy_engine_state" in checkpoint:
            privacy_state = checkpoint["privacy_engine_state"]
            logger.info(f"Resuming with privacy budget ε={privacy_state.get('epsilon', 'N/A')}")

            # Note: PrivacyEngine accountant state is restored during prepare_training()
            # We just log the information here for transparency
            if (
                self.privacy_engine is not None
                and privacy_state.get("accountant") is not None
                and hasattr(self.privacy_engine, "accountant")
            ):
                self.privacy_engine.accountant.load_state_dict(privacy_state["accountant"])
                logger.success("Restored PrivacyEngine accountant state")

        return epoch, best_acc, train_losses, val_losses, train_accs, val_accs

    # Override the base training loop to inject epsilon tracking after each epoch
    def train(
        self,
        start_epoch=0,
        resume_best_acc=0.0,
        resume_train_losses=None,
        resume_val_losses=None,
        resume_train_accs=None,
        resume_val_accs=None,
    ):
        """
        Main training loop with privacy budget tracking.

        Extends BaseTrainer.train() to compute and log epsilon after each epoch.

        Args:
            start_epoch: Epoch to start from (for resuming)
            resume_best_acc: Best accuracy from resumed checkpoint
            resume_train_losses: Training losses from resumed checkpoint
            resume_val_losses: Validation losses from resumed checkpoint
            resume_train_accs: Training accuracies from resumed checkpoint
            resume_val_accs: Validation accuracies from resumed checkpoint

        Returns:
            Tuple of (model, train_losses, val_losses, train_accs, val_accs)
        """
        import time

        since = time.time()

        # Initialize or resume metrics
        best_acc = resume_best_acc
        best_epoch = start_epoch if resume_best_acc > 0 else 0
        train_losses = resume_train_losses if resume_train_losses is not None else []
        val_losses = resume_val_losses if resume_val_losses is not None else []
        train_accs = resume_train_accs if resume_train_accs is not None else []
        val_accs = resume_val_accs if resume_val_accs is not None else []
        early_stop_triggered = False

        # Prepare for training (initializes PrivacyEngine)
        self.prepare_training()

        # Save initial checkpoint if starting fresh
        if start_epoch == 0:
            from ml_src.core.checkpointing import save_summary

            metrics = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            }
            self.save_checkpoint(0, best_acc, metrics, self.best_model_path)

        # Create initial summary
        from ml_src.core.checkpointing import save_summary

        save_summary(
            summary_path=self.summary_path,
            status="running",
            config=self.config,
            device=str(self.device),
            dataset_sizes=self.dataset_sizes,
            num_parameters=self.num_params,
            start_time=since,
            current_epoch=start_epoch,
            total_epochs=self.num_epochs,
            best_acc=best_acc if best_acc > 0 else None,
            best_epoch=best_epoch if best_acc > 0 else None,
        )

        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            logger.opt(colors=True).info(f"<yellow>Epoch {epoch}/{self.num_epochs - 1}</yellow>")
            logger.opt(colors=True).info("<dim>" + "-" * 50 + "</dim>")

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    if phase == "train":
                        outputs, loss = self.training_step(inputs, labels)
                    else:
                        with torch.no_grad():
                            outputs, loss = self.validation_step(inputs, labels)

                    # Get predictions
                    _, preds = torch.max(outputs, 1)

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    self.scheduler.step()
                    # Log learning rate to TensorBoard
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("Learning_Rate", current_lr, epoch)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                phase_color = "<blue>" if phase == "train" else "<magenta>"
                logger.opt(colors=True).info(
                    f"{phase_color}{phase}</> Loss: <yellow>{epoch_loss:.4f}</> "
                    f"Acc: <yellow>{epoch_acc:.4f}</>"
                )

                # Store metrics
                if phase == "train":
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc.item())
                    # Log to TensorBoard
                    self.writer.add_scalar("Loss/train", epoch_loss, epoch)
                    self.writer.add_scalar("Accuracy/train", epoch_acc.item(), epoch)
                else:
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc.item())
                    # Log to TensorBoard
                    self.writer.add_scalar("Loss/val", epoch_loss, epoch)
                    self.writer.add_scalar("Accuracy/val", epoch_acc.item(), epoch)

                # Save best model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    metrics = {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "train_accs": train_accs,
                        "val_accs": val_accs,
                    }
                    self.save_checkpoint(epoch, best_acc, metrics, self.best_model_path)
                    logger.success(f"New best model saved! Acc: {best_acc:.4f}")

                # Check early stopping
                if phase == "val" and self.early_stopping is not None:
                    metric_value = (
                        epoch_acc.item() if self.early_stopping.metric == "val_acc" else epoch_loss
                    )
                    if self.early_stopping.should_stop(epoch, metric_value):
                        # Save last checkpoint before stopping
                        metrics = {
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "train_accs": train_accs,
                            "val_accs": val_accs,
                        }
                        self.save_checkpoint(epoch, best_acc, metrics, self.last_model_path)
                        early_stop_triggered = True
                        logger.opt(colors=True).warning(
                            f"<yellow>Early stopping at epoch {epoch}</yellow>. "
                            f"Best accuracy: {best_acc:.4f} at epoch {best_epoch}"
                        )
                        break

            # Track privacy budget after each epoch
            self._track_privacy_budget(epoch)

            # Save last model checkpoint after each epoch
            metrics = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            }
            self.save_checkpoint(epoch, best_acc, metrics, self.last_model_path)

            # Update summary after each epoch
            save_summary(
                summary_path=self.summary_path,
                status="running",
                config=self.config,
                device=str(self.device),
                dataset_sizes=self.dataset_sizes,
                num_parameters=self.num_params,
                start_time=since,
                current_epoch=epoch + 1,
                total_epochs=self.num_epochs,
                best_acc=best_acc,
                best_epoch=best_epoch,
                final_train_acc=train_accs[-1] if train_accs else None,
                final_train_loss=train_losses[-1] if train_losses else None,
                final_val_acc=val_accs[-1] if val_accs else None,
                final_val_loss=val_losses[-1] if val_losses else None,
            )

            logger.info("")

            # Break outer loop if early stopping triggered
            if early_stop_triggered:
                break

        time_elapsed = time.time() - since
        end_time = time.time()

        # Log completion message
        if early_stop_triggered:
            logger.success(
                f"Training stopped early in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
        else:
            logger.success(
                f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
        logger.success(f"Best val Acc: {best_acc:.4f}")

        # Log final privacy budget
        try:
            final_epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
            logger.success(f"Final Privacy Budget: ε={final_epsilon:.2f} at δ={self.target_delta}")
        except (ValueError, RuntimeError):
            logger.warning("Could not compute final privacy budget (insufficient training steps)")

        # Load best model weights
        checkpoint = torch.load(self.best_model_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Generate metrics for train and val sets
        from ml_src.core.trainers.base import collect_predictions

        logger.info("Generating metrics for train and val sets...")

        # Collect predictions for train
        train_labels, train_preds = collect_predictions(
            self.model, self.dataloaders["train"], self.device
        )

        # Log to TensorBoard
        import os

        from ml_src.core.metrics import (
            get_classification_report_str,
            log_confusion_matrix_to_tensorboard,
            save_classification_report,
        )

        log_confusion_matrix_to_tensorboard(
            self.writer,
            train_labels,
            train_preds,
            self.class_names,
            "Confusion_Matrix/train",
            self.num_epochs - 1,
        )
        train_report = get_classification_report_str(train_labels, train_preds, self.class_names)
        self.writer.add_text("Classification_Report/train", train_report, self.num_epochs - 1)

        # Save to files
        save_classification_report(
            train_labels,
            train_preds,
            self.class_names,
            os.path.join(self.run_dir, "logs", "classification_report_train.txt"),
        )

        # Collect predictions for val
        val_labels, val_preds = collect_predictions(
            self.model, self.dataloaders["val"], self.device
        )

        # Log to TensorBoard
        log_confusion_matrix_to_tensorboard(
            self.writer,
            val_labels,
            val_preds,
            self.class_names,
            "Confusion_Matrix/val",
            self.num_epochs - 1,
        )
        val_report = get_classification_report_str(val_labels, val_preds, self.class_names)
        self.writer.add_text("Classification_Report/val", val_report, self.num_epochs - 1)

        # Save to files
        save_classification_report(
            val_labels,
            val_preds,
            self.class_names,
            os.path.join(self.run_dir, "logs", "classification_report_val.txt"),
        )

        # Save final summary
        if early_stop_triggered:
            stopped_epoch = self.early_stopping.stopped_epoch if self.early_stopping else epoch
            final_status = f"early_stopped_epoch_{stopped_epoch}"
        else:
            final_status = "completed"
        save_summary(
            summary_path=self.summary_path,
            status=final_status,
            config=self.config,
            device=str(self.device),
            dataset_sizes=self.dataset_sizes,
            num_parameters=self.num_params,
            start_time=since,
            end_time=end_time,
            current_epoch=self.num_epochs,
            total_epochs=self.num_epochs,
            best_acc=best_acc,
            best_epoch=best_epoch,
            final_train_acc=train_accs[-1] if train_accs else None,
            final_train_loss=train_losses[-1] if train_losses else None,
            final_val_acc=val_accs[-1] if val_accs else None,
            final_val_loss=val_losses[-1] if val_losses else None,
        )

        # Close TensorBoard writer
        self.writer.close()
        logger.info("TensorBoard writer closed")

        return self.model, train_losses, val_losses, train_accs, val_accs

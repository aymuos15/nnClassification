"""Training module for model training."""

import os
import time

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from ml_src.core.checkpointing import count_parameters, save_checkpoint, save_summary
from ml_src.core.metrics import (
    get_classification_report_str,
    log_confusion_matrix_to_tensorboard,
    save_classification_report,
)


def collect_predictions(model, dataloader, device):
    """
    Collect all predictions and labels from a dataloader.

    Args:
        model: The model to use
        dataloader: DataLoader to iterate through
        device: Device to run on

    Returns:
        Tuple of (all_labels, all_predictions)
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    device,
    config,
    run_dir,
    class_names,
    start_epoch=0,
    resume_best_acc=0.0,
    resume_train_losses=None,
    resume_val_losses=None,
    resume_train_accs=None,
    resume_val_accs=None,
):
    """
    Train the model.

    Args:
        model: The model to train
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        dataloaders: Dictionary of dataloaders for train and val
        dataset_sizes: Dictionary of dataset sizes
        device: Device to train on
        config: Configuration dictionary
        run_dir: Directory to save model checkpoints
        class_names: List of class names
        start_epoch: Epoch to start from (for resuming)
        resume_best_acc: Best accuracy from resumed checkpoint
        resume_train_losses: Training losses from resumed checkpoint
        resume_val_losses: Validation losses from resumed checkpoint
        resume_train_accs: Training accuracies from resumed checkpoint
        resume_val_accs: Validation accuracies from resumed checkpoint

    Returns:
        Trained model with best weights loaded, train_losses, val_losses, train_accs, val_accs
    """
    since = time.time()
    num_epochs = config["training"]["num_epochs"]

    best_model_path = os.path.join(run_dir, "weights", "best.pt")
    last_model_path = os.path.join(run_dir, "weights", "last.pt")
    summary_path = os.path.join(run_dir, "summary.txt")

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)
    logger.info(f"TensorBoard logs: {tensorboard_dir}")

    # Initialize or resume metrics
    best_acc = resume_best_acc
    best_epoch = start_epoch if resume_best_acc > 0 else 0
    train_losses = resume_train_losses if resume_train_losses is not None else []
    val_losses = resume_val_losses if resume_val_losses is not None else []
    train_accs = resume_train_accs if resume_train_accs is not None else []
    val_accs = resume_val_accs if resume_val_accs is not None else []

    # Count model parameters
    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} trainable parameters")

    # Save initial checkpoint if starting fresh
    if start_epoch == 0:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            0,
            best_acc,
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            config,
            best_model_path,
        )

    # Create initial summary
    save_summary(
        summary_path=summary_path,
        status="running",
        config=config,
        device=str(device),
        dataset_sizes=dataset_sizes,
        num_parameters=num_params,
        start_time=since,
        current_epoch=start_epoch,
        total_epochs=num_epochs,
        best_acc=best_acc if best_acc > 0 else None,
        best_epoch=best_epoch if best_acc > 0 else None,
    )

    for epoch in range(start_epoch, num_epochs):
        logger.opt(colors=True).info(f"<yellow>Epoch {epoch}/{num_epochs - 1}</yellow>")
        logger.opt(colors=True).info("<dim>" + "-" * 50 + "</dim>")

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()
                # Log learning rate to TensorBoard
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("Learning_Rate", current_lr, epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            phase_color = "<blue>" if phase == "train" else "<magenta>"
            logger.opt(colors=True).info(
                f"{phase_color}{phase}</> Loss: <yellow>{epoch_loss:.4f}</> Acc: <yellow>{epoch_acc:.4f}</>"
            )

            # Store metrics
            if phase == "train":
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
                # Log to TensorBoard
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("Accuracy/train", epoch_acc.item(), epoch)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                # Log to TensorBoard
                writer.add_scalar("Loss/val", epoch_loss, epoch)
                writer.add_scalar("Accuracy/val", epoch_acc.item(), epoch)

            # Save best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_acc,
                    train_losses,
                    val_losses,
                    train_accs,
                    val_accs,
                    config,
                    best_model_path,
                )
                logger.success(f"New best model saved! Acc: {best_acc:.4f}")

        # Save last model checkpoint after each epoch
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_acc,
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            config,
            last_model_path,
        )

        # Update summary after each epoch
        save_summary(
            summary_path=summary_path,
            status="running",
            config=config,
            device=str(device),
            dataset_sizes=dataset_sizes,
            num_parameters=num_params,
            start_time=since,
            current_epoch=epoch + 1,
            total_epochs=num_epochs,
            best_acc=best_acc,
            best_epoch=best_epoch,
            final_train_acc=train_accs[-1] if train_accs else None,
            final_train_loss=train_losses[-1] if train_losses else None,
            final_val_acc=val_accs[-1] if val_accs else None,
            final_val_loss=val_losses[-1] if val_losses else None,
        )

        logger.info("")

    time_elapsed = time.time() - since
    end_time = time.time()
    logger.success(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    logger.success(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Generate metrics for train and val sets
    logger.info("Generating metrics for train and val sets...")

    # Collect predictions for train
    train_labels, train_preds = collect_predictions(model, dataloaders["train"], device)

    # Log to TensorBoard
    log_confusion_matrix_to_tensorboard(
        writer,
        train_labels,
        train_preds,
        class_names,
        "Confusion_Matrix/train",
        num_epochs - 1,
    )
    train_report = get_classification_report_str(train_labels, train_preds, class_names)
    writer.add_text("Classification_Report/train", train_report, num_epochs - 1)

    # Save to files (for backward compatibility)
    save_classification_report(
        train_labels,
        train_preds,
        class_names,
        os.path.join(run_dir, "logs", "classification_report_train.txt"),
    )

    # Collect predictions for val
    val_labels, val_preds = collect_predictions(model, dataloaders["val"], device)

    # Log to TensorBoard
    log_confusion_matrix_to_tensorboard(
        writer,
        val_labels,
        val_preds,
        class_names,
        "Confusion_Matrix/val",
        num_epochs - 1,
    )
    val_report = get_classification_report_str(val_labels, val_preds, class_names)
    writer.add_text("Classification_Report/val", val_report, num_epochs - 1)

    # Save to files (for backward compatibility)
    save_classification_report(
        val_labels,
        val_preds,
        class_names,
        os.path.join(run_dir, "logs", "classification_report_val.txt"),
    )

    # Save final completed summary
    save_summary(
        summary_path=summary_path,
        status="completed",
        config=config,
        device=str(device),
        dataset_sizes=dataset_sizes,
        num_parameters=num_params,
        start_time=since,
        end_time=end_time,
        current_epoch=num_epochs,
        total_epochs=num_epochs,
        best_acc=best_acc,
        best_epoch=best_epoch,
        final_train_acc=train_accs[-1] if train_accs else None,
        final_train_loss=train_losses[-1] if train_losses else None,
        final_val_acc=val_accs[-1] if val_accs else None,
        final_val_loss=val_losses[-1] if val_losses else None,
    )

    # Close TensorBoard writer
    writer.close()
    logger.info("TensorBoard writer closed")

    return model, train_losses, val_losses, train_accs, val_accs

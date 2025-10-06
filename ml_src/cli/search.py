#!/usr/bin/env python3
"""
Hyperparameter search script using Optuna.
"""

import argparse
import os

import torch
from loguru import logger

from ml_src.core.config import load_config, override_config
from ml_src.core.data import get_class_names, get_datasets
from ml_src.core.loader import get_dataloaders, get_dataset_sizes
from ml_src.core.logging import setup_logging
from ml_src.core.loss import get_criterion
from ml_src.core.network import get_model
from ml_src.core.optimizer import get_optimizer, get_scheduler
from ml_src.core.seeding import set_seed
from ml_src.core.trainers import get_trainer


def run_trial(trial_config, trial_number, trial=None):
    """
    Run a single training trial with given configuration.

    Args:
        trial_config: Configuration for this trial
        trial_number: Trial number
        trial: Optuna trial object (for pruning)

    Returns:
        Dictionary with validation metrics
    """
    # Set seed for reproducibility
    seed = trial_config.get("seed", 42)
    deterministic = trial_config.get("deterministic", False)
    set_seed(seed, deterministic)

    # Determine device
    device_str = trial_config["training"]["device"]
    if device_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")

    logger.info(f"Trial {trial_number}: Using device: {device}")

    # Create datasets and dataloaders
    datasets = get_datasets(trial_config)
    class_names = get_class_names(datasets)
    dataloaders = get_dataloaders(datasets, trial_config)
    dataset_sizes = get_dataset_sizes(datasets)

    # Create model
    model = get_model(trial_config, device)

    # Create criterion, optimizer, scheduler
    criterion = get_criterion()
    optimizer = get_optimizer(model, trial_config)
    scheduler = get_scheduler(optimizer, trial_config)

    # Create trial-specific run directory
    search_config = trial_config["search"]
    study_name = search_config["study_name"]
    trial_run_dir = os.path.join("runs", "optuna_studies", study_name, f"trial_{trial_number}")
    os.makedirs(trial_run_dir, exist_ok=True)
    os.makedirs(os.path.join(trial_run_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(trial_run_dir, "logs"), exist_ok=True)

    # Setup logging for this trial
    setup_logging(trial_run_dir, filename="train.log")
    logger.info(f"Trial {trial_number} run directory: {trial_run_dir}")

    # Create trainer
    trainer = get_trainer(
        config=trial_config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        run_dir=trial_run_dir,
        class_names=class_names,
    )

    # Store trial reference for pruning
    if trial is not None:
        trainer.optuna_trial = trial

    # Train the model
    try:
        model, train_losses, val_losses, train_accs, val_accs = trainer.train()
    except Exception as e:
        # Handle pruned trials
        import optuna

        if isinstance(e, optuna.TrialPruned):
            logger.info(f"Trial {trial_number} pruned")
            raise
        else:
            logger.error(f"Trial {trial_number} failed with error: {e}")
            raise

    # Return final validation metrics
    metric_name = search_config.get("metric", "val_acc")
    if metric_name == "val_acc":
        return {"val_acc": val_accs[-1], "val_loss": val_losses[-1]}
    else:
        return {"val_acc": val_accs[-1], "val_loss": val_losses[-1]}


def main():
    """Main function for hyperparameter search."""
    # Check if optuna is installed
    try:
        import optuna  # noqa: F401
    except ImportError:
        logger.error(
            "Optuna is not installed. Install with:\n"
            "  pip install -e '.[optuna]'\n"
            "Or generate a config with search pre-configured:\n"
            "  ml-init-config data/my_dataset --optuna"
        )
        return

    parser = argparse.ArgumentParser(
        description="Hyperparameter search for image classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run hyperparameter search
  ml-search --config configs/my_config.yaml

  # Specify number of trials
  ml-search --config configs/my_config.yaml --n-trials 100

  # Resume existing study
  ml-search --config configs/my_config.yaml --resume

  # Override study name
  ml-search --config configs/my_config.yaml --study-name my_custom_study
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file with search section",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Number of trials to run (overrides config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds (overrides config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing study instead of creating new one",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="Override study name from config",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Fold number for cross-validation (0-indexed, default: 0)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    logger.info("=" * 70)
    logger.info("Hyperparameter Search")
    logger.info("=" * 70)

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    config, overrides = override_config(config, args)

    # Check if config has search section
    if "search" not in config:
        logger.error(
            "Config file does not contain 'search' section.\n"
            "Generate a new config with search support:\n"
            "  ml-init-config data/my_dataset --optuna\n"
            "Or manually add search section to your config."
        )
        return

    search_config = config["search"]

    # Override settings from command line
    if args.n_trials:
        search_config["n_trials"] = args.n_trials
    if args.timeout:
        search_config["timeout"] = args.timeout
    if args.study_name:
        search_config["study_name"] = args.study_name

    logger.info(f"Study name: {search_config['study_name']}")
    logger.info(f"Storage: {search_config.get('storage', 'in-memory')}")
    logger.info(f"Number of trials: {search_config['n_trials']}")
    logger.info(f"Direction: {search_config.get('direction', 'maximize')}")
    logger.info(f"Metric: {search_config.get('metric', 'val_acc')}")

    # Display search space
    logger.info("\nSearch space:")
    for param_name, param_config in search_config["search_space"].items():
        if param_config["type"] == "categorical":
            logger.info(f"  {param_name}: {param_config['choices']}")
        elif param_config["type"] in ["uniform", "loguniform"]:
            logger.info(
                f"  {param_name}: [{param_config['low']}, {param_config['high']}] "
                f"({param_config['type']})"
            )
        elif param_config["type"] == "int":
            logger.info(f"  {param_name}: [{param_config['low']}, {param_config['high']}] (int)")

    # Import search utilities
    from ml_src.core.search import create_objective, create_study, save_best_config

    # Create study
    try:
        study = create_study(search_config, resume=args.resume)
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.error(
                f"Study '{search_config['study_name']}' already exists. "
                "Use --resume to continue or change --study-name"
            )
        else:
            logger.error(f"Failed to create study: {e}")
        return

    # Create objective function
    def run_training_wrapper(trial_config, trial_number, trial):
        return run_trial(trial_config, trial_number, trial)

    objective = create_objective(config, search_config, run_training_wrapper)

    # Run optimization
    logger.info("\n" + "=" * 70)
    logger.info("Starting Optimization")
    logger.info("=" * 70)

    try:
        study.optimize(
            objective,
            n_trials=search_config["n_trials"],
            timeout=search_config.get("timeout"),
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("Optimization Complete")
    logger.info("=" * 70)

    logger.success(f"Number of finished trials: {len(study.trials)}")
    logger.success(f"Best trial: {study.best_trial.number}")
    logger.success(f"Best value: {study.best_value:.4f}")

    logger.info("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Save best configuration
    study_dir = os.path.join("runs", "optuna_studies", search_config["study_name"])
    os.makedirs(study_dir, exist_ok=True)
    best_config_path = os.path.join(study_dir, "best_config.yaml")

    save_best_config(study, config, best_config_path)

    # Print next steps
    logger.info("\n" + "=" * 70)
    logger.info("Next Steps")
    logger.info("=" * 70)
    logger.info(
        f"1. View results: ml-visualise --mode search --study-name {search_config['study_name']}"
    )
    logger.info(f"2. Train with best config: ml-train --config {best_config_path}")
    logger.info(f"3. Review trial logs: ls {study_dir}/trial_*/logs/")


if __name__ == "__main__":
    main()

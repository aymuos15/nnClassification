"""Hyperparameter search utilities using Optuna."""

import copy
import os
from typing import Any, Dict, Optional

from loguru import logger


def suggest_hyperparameter(trial, param_name: str, param_config: Dict[str, Any]) -> Any:
    """
    Suggest a hyperparameter value based on its configuration.

    Args:
        trial: Optuna trial object
        param_name: Name of the parameter
        param_config: Configuration dict with 'type' and range information

    Returns:
        Suggested parameter value

    Raises:
        ValueError: If parameter type is not supported

    Example:
        >>> param_config = {'type': 'loguniform', 'low': 1e-5, 'high': 1e-1}
        >>> lr = suggest_hyperparameter(trial, 'optimizer.lr', param_config)
    """
    param_type = param_config["type"]

    if param_type == "categorical":
        return trial.suggest_categorical(param_name, param_config["choices"])
    elif param_type == "uniform":
        return trial.suggest_float(param_name, param_config["low"], param_config["high"])
    elif param_type == "loguniform":
        return trial.suggest_float(param_name, param_config["low"], param_config["high"], log=True)
    elif param_type == "int":
        return trial.suggest_int(param_name, param_config["low"], param_config["high"])
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")


def apply_hyperparameter(config: Dict, param_path: str, value: Any) -> None:
    """
    Apply a hyperparameter value to nested config dictionary.

    Args:
        config: Configuration dictionary to modify
        param_path: Dot-separated path to parameter (e.g., 'optimizer.lr')
        value: Value to set

    Example:
        >>> config = {'optimizer': {'lr': 0.001}}
        >>> apply_hyperparameter(config, 'optimizer.lr', 0.01)
        >>> config['optimizer']['lr']
        0.01
    """
    keys = param_path.split(".")
    current = config

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the final value
    current[keys[-1]] = value


def create_trial_config(base_config: Dict, trial, search_space: Dict[str, Dict[str, Any]]) -> Dict:
    """
    Create a trial-specific configuration by suggesting hyperparameters.

    Args:
        base_config: Base configuration dictionary
        trial: Optuna trial object
        search_space: Dictionary mapping parameter paths to their configurations

    Returns:
        Trial-specific configuration dictionary

    Example:
        >>> search_space = {
        ...     'optimizer.lr': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-1}
        ... }
        >>> trial_config = create_trial_config(base_config, trial, search_space)
    """
    # Deep copy base config to avoid modifying original
    trial_config = copy.deepcopy(base_config)

    # Suggest and apply each hyperparameter
    for param_name, param_config in search_space.items():
        value = suggest_hyperparameter(trial, param_name, param_config)
        apply_hyperparameter(trial_config, param_name, value)
        logger.info(f"Trial {trial.number}: {param_name} = {value}")

    return trial_config


def create_objective(
    base_config: Dict,
    search_config: Dict,
    run_training_func,
):
    """
    Create an objective function for Optuna optimization.

    Args:
        base_config: Base configuration dictionary
        search_config: Search configuration with search_space, metric, etc.
        run_training_func: Function that trains model and returns metrics
            Should have signature: (config, trial_number, trial) -> Dict[str, float]

    Returns:
        Objective function for Optuna

    Example:
        >>> def run_training(config, trial_num, trial):
        ...     # Train model with config
        ...     return {'val_acc': 0.95, 'val_loss': 0.1}
        >>> objective = create_objective(base_config, search_config, run_training)
        >>> study.optimize(objective, n_trials=10)
    """

    def objective(trial):
        """Objective function for a single trial."""
        # Create trial-specific config
        trial_config = create_trial_config(base_config, trial, search_config["search_space"])

        # Run training and get results
        results = run_training_func(trial_config, trial.number, trial)

        # Extract metric to optimize
        metric_name = search_config.get("metric", "val_acc")
        metric_value = results.get(metric_name)

        if metric_value is None:
            raise ValueError(
                f"Metric '{metric_name}' not found in results. Available: {list(results.keys())}"
            )

        logger.info(f"Trial {trial.number} completed: {metric_name} = {metric_value:.4f}")

        return metric_value

    return objective


def create_sampler(sampler_config: Dict):
    """
    Create an Optuna sampler from configuration.

    Args:
        sampler_config: Sampler configuration dictionary

    Returns:
        Optuna sampler instance

    Example:
        >>> config = {'type': 'TPESampler', 'n_startup_trials': 10}
        >>> sampler = create_sampler(config)
    """
    try:
        import optuna
    except ImportError as err:
        raise ImportError("Optuna is not installed. Install with: pip install -e '.[optuna]'") from err

    sampler_type = sampler_config.get("type", "TPESampler")

    if sampler_type == "TPESampler":
        return optuna.samplers.TPESampler(
            n_startup_trials=sampler_config.get("n_startup_trials", 10)
        )
    elif sampler_type == "RandomSampler":
        return optuna.samplers.RandomSampler()
    elif sampler_type == "GridSampler":
        # Grid sampler requires explicit search space
        raise NotImplementedError("GridSampler requires explicit search space mapping")
    elif sampler_type == "CmaEsSampler":
        return optuna.samplers.CmaEsSampler(
            n_startup_trials=sampler_config.get("n_startup_trials", 1)
        )
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


def create_pruner(pruner_config: Optional[Dict]):
    """
    Create an Optuna pruner from configuration.

    Args:
        pruner_config: Pruner configuration dictionary or None

    Returns:
        Optuna pruner instance or None

    Example:
        >>> config = {'type': 'MedianPruner', 'n_warmup_steps': 5}
        >>> pruner = create_pruner(config)
    """
    if pruner_config is None:
        return None

    try:
        import optuna
    except ImportError as err:
        raise ImportError("Optuna is not installed. Install with: pip install -e '.[optuna]'") from err

    pruner_type = pruner_config.get("type", "MedianPruner")

    if pruner_type == "MedianPruner":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner_config.get("n_startup_trials", 5),
            n_warmup_steps=pruner_config.get("n_warmup_steps", 5),
        )
    elif pruner_type == "PercentilePruner":
        return optuna.pruners.PercentilePruner(
            percentile=pruner_config.get("percentile", 25.0),
            n_startup_trials=pruner_config.get("n_startup_trials", 5),
            n_warmup_steps=pruner_config.get("n_warmup_steps", 5),
        )
    elif pruner_type == "HyperbandPruner":
        return optuna.pruners.HyperbandPruner(
            min_resource=pruner_config.get("min_resource", 1),
            max_resource=pruner_config.get("max_resource", "auto"),
        )
    else:
        raise ValueError(f"Unsupported pruner type: {pruner_type}")


def create_study(search_config: Dict, resume: bool = False):
    """
    Create or load an Optuna study.

    Args:
        search_config: Search configuration dictionary
        resume: Whether to resume existing study or create new one

    Returns:
        Optuna study object

    Example:
        >>> config = {
        ...     'study_name': 'my_study',
        ...     'storage': 'sqlite:///optuna.db',
        ...     'direction': 'maximize'
        ... }
        >>> study = create_study(config, resume=False)
    """
    try:
        import optuna
    except ImportError as err:
        raise ImportError("Optuna is not installed. Install with: pip install -e '.[optuna]'") from err

    study_name = search_config["study_name"]
    storage = search_config.get("storage")
    direction = search_config.get("direction", "maximize")

    # Create sampler and pruner
    sampler = create_sampler(search_config.get("sampler", {}))
    pruner = create_pruner(search_config.get("pruner"))

    if resume:
        logger.info(f"Loading existing study: {study_name}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )
    else:
        logger.info(f"Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False,
        )

    return study


def save_best_config(study, base_config: Dict, output_path: str) -> None:
    """
    Save the best configuration to a YAML file.

    Args:
        study: Optuna study object
        base_config: Base configuration dictionary
        output_path: Path to save the best configuration

    Example:
        >>> save_best_config(study, base_config, 'runs/study/best_config.yaml')
    """
    import yaml

    # Get best trial
    best_trial = study.best_trial
    logger.success(f"Best trial: {best_trial.number}")
    logger.success(f"Best value: {best_trial.value:.4f}")

    # Create config with best parameters
    best_config = copy.deepcopy(base_config)

    for param_name, param_value in best_trial.params.items():
        apply_hyperparameter(best_config, param_name, param_value)
        logger.info(f"Best {param_name} = {param_value}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save config
    with open(output_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    logger.success(f"Best configuration saved to: {output_path}")

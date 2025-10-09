"""Flower client implementation that wraps existing trainers."""

import os
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import torch
from loguru import logger

from ml_src.core.data.datasets import get_datasets
from ml_src.core.loader import get_dataloaders, get_dataset_sizes
from ml_src.core.losses import get_criterion
from ml_src.core.network import get_model
from ml_src.core.optimizer import get_optimizer, get_scheduler
from ml_src.core.trainers import get_trainer

try:
    import flwr as fl
    from flwr.common import NDArrays, Scalar
except ImportError:
    raise ImportError(
        "Flower is required for federated learning. Install with:\n"
        "  uv pip install -e '.[flower]'\n"
        "Or:\n"
        "  pip install flwr>=1.7.0"
    )


def get_parameters(model: torch.nn.Module) -> NDArrays:
    """
    Extract model parameters as a list of NumPy arrays.

    Args:
        model: PyTorch model

    Returns:
        List of NumPy arrays containing model parameters
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: NDArrays) -> None:
    """
    Set model parameters from a list of NumPy arrays.

    Args:
        model: PyTorch model
        parameters: List of NumPy arrays containing model parameters
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client that wraps existing ML classifier trainers.

    This client enables federated learning by composing with existing trainer
    infrastructure (standard, mixed_precision, dp, accelerate). Each client can
    use a different trainer type based on its capabilities and requirements.

    Args:
        config: Configuration dictionary (same format as standalone training)
        client_id: Unique identifier for this client
        run_dir: Directory for client-specific logs and checkpoints

    Example:
        >>> config = load_config('configs/federated_experiment.yaml')
        >>> client = FlowerClient(config=config, client_id=0, run_dir='runs/fl_client_0')
        >>> # Flower will call fit() and evaluate() automatically
    """

    def __init__(self, config: Dict, client_id: int, run_dir: str):
        self.config = config
        self.client_id = client_id
        self.run_dir = run_dir

        # Setup device
        device_str = config.get("training", {}).get("device", "cuda:0")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        logger.info(f"[Client {client_id}] Initializing on device: {self.device}")

        # Create client-specific run directory
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        # Load client-specific data
        logger.info(f"[Client {client_id}] Loading client-specific data partition")
        self.datasets = self._load_client_data()
        self.dataloaders = get_dataloaders(self.datasets, config)
        self.dataset_sizes = get_dataset_sizes(self.datasets)

        # Initialize model, criterion, optimizer, scheduler
        logger.info(f"[Client {client_id}] Initializing model and training components")
        self.class_names = self.datasets["train"].classes
        num_classes = len(self.class_names)

        self.model = get_model(config, num_classes, self.device)
        self.criterion = get_criterion(config)
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = get_scheduler(config, self.optimizer)

        # Get trainer type from config (supports heterogeneous clients!)
        trainer_type = config.get("training", {}).get("trainer_type", "standard")
        logger.info(f"[Client {client_id}] Using trainer type: {trainer_type}")

        # Initialize the appropriate trainer (composition, not inheritance!)
        self.trainer = get_trainer(
            config=config,
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dataloaders=self.dataloaders,
            dataset_sizes=self.dataset_sizes,
            device=self.device,
            run_dir=run_dir,
            class_names=self.class_names,
        )

        logger.success(f"[Client {client_id}] Initialization complete")

    def _load_client_data(self) -> Dict:
        """
        Load client-specific data partition.

        Returns:
            Dictionary of datasets (train, val, test)
        """
        # Update config to point to client-specific data partitions
        client_config = self.config.copy()

        # Get data directory
        data_dir = client_config["data"]["data_dir"]
        splits_dir = os.path.join(data_dir, "splits")

        # Client-specific index files (e.g., client_0_train.txt, client_0_val.txt)
        train_index = os.path.join(splits_dir, f"client_{self.client_id}_train.txt")
        val_index = os.path.join(splits_dir, f"client_{self.client_id}_val.txt")
        test_index = os.path.join(splits_dir, "test.txt")  # Shared test set

        # Verify index files exist
        if not os.path.exists(train_index):
            raise FileNotFoundError(
                f"Client {self.client_id} train index not found: {train_index}\n"
                f"Run: ml-split --raw_data {data_dir}/raw --federated --num-clients N"
            )
        if not os.path.exists(val_index):
            raise FileNotFoundError(
                f"Client {self.client_id} val index not found: {val_index}\n"
                f"Run: ml-split --raw_data {data_dir}/raw --federated --num-clients N"
            )

        logger.info(f"[Client {self.client_id}] Loading data from {splits_dir}")

        # Load datasets using existing infrastructure
        datasets = get_datasets(
            data_dir=data_dir,
            transform_config=client_config.get("transforms", {}),
            train_index=train_index,
            val_index=val_index,
            test_index=test_index,
        )

        logger.info(
            f"[Client {self.client_id}] Data loaded: "
            f"train={len(datasets['train'])}, val={len(datasets['val'])}, "
            f"test={len(datasets['test'])}"
        )

        return datasets

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on the client's local data.

        This method is called by the Flower server during each federated round.
        It loads global parameters, trains locally using the configured trainer,
        and returns updated parameters.

        Args:
            parameters: Global model parameters from server
            config: Configuration dictionary from server (e.g., current round number)

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        logger.info(f"[Client {self.client_id}] Starting local training")

        # Load global parameters into local model
        set_parameters(self.model, parameters)

        # Get local epochs from server config (or use client's default)
        local_epochs = config.get("local_epochs", self.config["training"]["num_epochs"])

        # Temporarily update config for this training round
        original_num_epochs = self.config["training"]["num_epochs"]
        self.config["training"]["num_epochs"] = local_epochs
        self.trainer.num_epochs = local_epochs

        # Train locally using existing trainer infrastructure
        # This supports ALL trainer types: standard, mixed_precision, dp, accelerate!
        trained_model, train_losses, val_losses, train_accs, val_accs = self.trainer.train()

        # Restore original config
        self.config["training"]["num_epochs"] = original_num_epochs
        self.trainer.num_epochs = original_num_epochs

        # Extract updated parameters
        updated_parameters = get_parameters(self.model)

        # Prepare metrics to return to server
        metrics = {
            "train_loss": train_losses[-1] if train_losses else 0.0,
            "train_acc": train_accs[-1] if train_accs else 0.0,
            "val_loss": val_losses[-1] if val_losses else 0.0,
            "val_acc": val_accs[-1] if val_accs else 0.0,
        }

        logger.success(
            f"[Client {self.client_id}] Training complete: "
            f"val_acc={metrics['val_acc']:.4f}, val_loss={metrics['val_loss']:.4f}"
        )

        # Return updated parameters, number of training examples, and metrics
        return updated_parameters, self.dataset_sizes["train"], metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on the client's local validation data.

        Args:
            parameters: Global model parameters from server
            config: Configuration dictionary from server

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        logger.info(f"[Client {self.client_id}] Starting local evaluation")

        # Load global parameters into local model
        set_parameters(self.model, parameters)

        # Evaluate on local validation set
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in self.dataloaders["val"]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / self.dataset_sizes["val"]
        val_acc = (running_corrects.double() / self.dataset_sizes["val"]).item()

        logger.info(
            f"[Client {self.client_id}] Evaluation complete: "
            f"val_acc={val_acc:.4f}, val_loss={val_loss:.4f}"
        )

        metrics = {"val_acc": val_acc}

        return val_loss, self.dataset_sizes["val"], metrics


def create_client_fn(
    config: Dict, run_base_dir: str = "runs"
) -> Callable[[str], FlowerClient]:
    """
    Factory function to create Flower client instances.

    This function is used by Flower's simulation mode to create multiple clients.
    Each client gets its own unique ID and run directory.

    Args:
        config: Configuration dictionary
        run_base_dir: Base directory for client run directories

    Returns:
        Function that creates FlowerClient instances given a client ID

    Example:
        >>> config = load_config('configs/federated_config.yaml')
        >>> client_fn = create_client_fn(config)
        >>> # Flower calls client_fn(cid) to create clients
    """

    def client_fn(cid: str) -> FlowerClient:
        """Create a FlowerClient instance for given client ID."""
        client_id = int(cid)

        # Apply client-specific config overrides (from profiles or manifest)
        client_config = _apply_client_overrides(config, client_id)

        # Create client-specific run directory
        run_dir = os.path.join(run_base_dir, f"fl_client_{client_id}")

        return FlowerClient(config=client_config, client_id=client_id, run_dir=run_dir)

    return client_fn


def _apply_client_overrides(config: Dict, client_id: int) -> Dict:
    """
    Apply client-specific configuration overrides.

    Supports two modes:
    1. Profiles (simulation): Clients grouped by capability
    2. Manifest (deployment): Explicit per-client overrides

    Args:
        config: Base configuration dictionary
        client_id: Client identifier

    Returns:
        Configuration with client-specific overrides applied
    """
    import copy

    client_config = copy.deepcopy(config)

    # Check for client profiles (simulation mode)
    profiles = config.get("federated", {}).get("clients", {}).get("profiles", [])
    for profile in profiles:
        if client_id in profile.get("id", []):
            # Apply profile overrides
            for key, value in profile.items():
                if key != "id":
                    # Handle nested config keys (e.g., 'trainer_type' -> training.trainer_type)
                    if key in ["trainer_type", "batch_size", "device"]:
                        client_config["training"][key] = value
                    elif key == "dp":
                        client_config["training"]["dp"] = value
                    else:
                        client_config[key] = value

            logger.info(f"[Client {client_id}] Applied profile overrides: {profile}")
            break

    # Check for client manifest (deployment mode)
    manifest = config.get("federated", {}).get("clients", {}).get("manifest", [])
    for entry in manifest:
        if entry.get("id") == client_id:
            # Apply manifest overrides
            config_override = entry.get("config_override")
            if config_override and os.path.exists(config_override):
                import yaml

                with open(config_override) as f:
                    override = yaml.safe_load(f)

                # Deep merge override into client_config
                client_config = _deep_merge(client_config, override)

                logger.info(
                    f"[Client {client_id}] Applied manifest override from {config_override}"
                )
            break

    return client_config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge override dictionary into base dictionary.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    import copy

    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result

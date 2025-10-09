"""Ensemble inference strategy for combining multiple models."""

import os

import torch
from loguru import logger

from ml_src.core.config import load_config
from ml_src.core.inference.base import BaseInferenceStrategy
from ml_src.core.metrics.segmentation import calculate_iou
from ml_src.core.network import get_model, load_model
from ml_src.core.run import get_run_dir_from_checkpoint


class EnsembleInference(BaseInferenceStrategy):
    """
    Ensemble inference strategy that combines predictions from multiple models.

    This strategy loads multiple model checkpoints (e.g., from different CV folds
    or training runs) and aggregates their predictions using soft voting, hard
    voting, or weighted averaging.

    Example:
        >>> # Config with ensemble
        >>> config = {
        ...     'inference': {
        ...         'strategy': 'ensemble',
        ...         'ensemble': {
        ...             'checkpoints': [
        ...                 'runs/fold_0/weights/best.pt',
        ...                 'runs/fold_1/weights/best.pt',
        ...                 'runs/fold_2/weights/best.pt'
        ...             ],
        ...             'aggregation': 'soft_voting',
        ...             'weights': [1.0, 1.0, 1.0]
        ...         }
        ...     }
        ... }
        >>> strategy = EnsembleInference(
        ...     checkpoints=['runs/fold_0/weights/best.pt', ...],
        ...     config=config,
        ...     device='cuda:0',
        ...     aggregation='soft_voting'
        ... )
        >>> test_acc, results = strategy.run_inference(
        ...     model=None,  # Not used for ensemble
        ...     dataloader=test_loader,
        ...     dataset_size=100,
        ...     device=device,
        ...     class_names=['cat', 'dog']
        ... )
    """

    def __init__(
        self,
        checkpoints,
        config,
        device,
        aggregation="soft_voting",
        weights=None,
        use_ema: bool = False,
    ):
        """
        Initialize ensemble inference strategy.

        Args:
            checkpoints: List of checkpoint paths to load
            config: Configuration dictionary (needed to recreate models)
            device: Device to run inference on
            aggregation: Method to aggregate predictions
                - 'soft_voting': Average logits (default, recommended)
                - 'hard_voting': Majority vote on predicted classes
                - 'weighted': Weighted average of logits using provided weights
            weights: List of weights for each model (only used with 'weighted' aggregation)
                If None and aggregation='weighted', uses equal weights
        """
        if not checkpoints or len(checkpoints) == 0:
            raise ValueError("Must provide at least one checkpoint for ensemble")

        self.checkpoints = checkpoints
        self.config = config
        self.device = device
        self.aggregation = aggregation
        self.weights = weights
        self.use_ema = use_ema

        # Validate aggregation method
        valid_methods = {"soft_voting", "hard_voting", "weighted"}
        if aggregation not in valid_methods:
            raise ValueError(
                f"Unknown aggregation method: '{aggregation}'. "
                f"Valid options: {sorted(valid_methods)}"
            )

        # Validate weights for weighted aggregation
        if aggregation == "weighted":
            if weights is None:
                # Default to equal weights
                self.weights = [1.0 / len(checkpoints)] * len(checkpoints)
                logger.info("No weights provided, using equal weights")
            elif len(weights) != len(checkpoints):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of checkpoints ({len(checkpoints)})"
                )
            else:
                # Normalize weights to sum to 1
                weight_sum = sum(weights)
                self.weights = [w / weight_sum for w in weights]

        logger.info(f"Ensemble with {len(checkpoints)} models")
        logger.info(f"Aggregation: {aggregation}")
        if aggregation == "weighted":
            logger.info(f"Weights: {self.weights}")

        # Load all models
        self.models = self._load_models()

    def _load_models(self):
        """Load all models from checkpoints."""
        models = []

        for i, checkpoint_path in enumerate(self.checkpoints):
            logger.info(f"Loading model {i + 1}/{len(self.checkpoints)}: {checkpoint_path}")

            # Validate checkpoint exists
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Ensure checkpoint model configuration matches current configuration
            self._validate_checkpoint_config(checkpoint_path)

            # Create model architecture
            model = get_model(self.config, self.device)

            # Load checkpoint
            model = load_model(model, checkpoint_path, self.device, use_ema=self.use_ema)

            # Set to eval mode
            model.eval()

            models.append(model)

        logger.success(f"Loaded {len(models)} models successfully")
        return models

    @staticmethod
    def _model_signature(model_cfg):
        return (
            model_cfg.get("type", "base"),
            model_cfg.get("architecture"),
            model_cfg.get("custom_architecture"),
            model_cfg.get("num_classes"),
        )

    def _validate_checkpoint_config(self, checkpoint_path):
        run_dir = get_run_dir_from_checkpoint(checkpoint_path)
        config_path = os.path.join(run_dir, "config.yaml")

        if not os.path.exists(config_path):
            logger.warning(
                "Config file not found for checkpoint {}. Skipping compatibility check.",
                checkpoint_path,
            )
            return

        try:
            checkpoint_config = load_config(config_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load config for checkpoint {} ({}). Skipping compatibility check.",
                checkpoint_path,
                exc,
            )
            return

        expected_signature = self._model_signature(self.config.get("model", {}))
        checkpoint_signature = self._model_signature(checkpoint_config.get("model", {}))

        if checkpoint_signature != expected_signature:
            raise ValueError(
                "Checkpoint model mismatch for {}. Expected {} but found {}. "
                "Ensure all ensemble checkpoints share the same model architecture and class count.".format(
                    checkpoint_path, expected_signature, checkpoint_signature
                )
            )

    def _aggregate_logits(self, all_logits):
        """
        Aggregate logits from multiple models.

        Args:
            all_logits: List of logit tensors from each model
                Each tensor shape: (batch_size, num_classes)

        Returns:
            Aggregated logits tensor (batch_size, num_classes)
        """
        if self.aggregation == "soft_voting":
            # Average logits
            return torch.stack(all_logits).mean(dim=0)

        elif self.aggregation == "weighted":
            # Weighted average of logits
            weighted = torch.stack(
                [logits * weight for logits, weight in zip(all_logits, self.weights)]
            )
            return weighted.sum(dim=0)

        elif self.aggregation == "hard_voting":
            # Majority vote on predicted classes
            class_preds = torch.stack([logits.argmax(dim=1) for logits in all_logits])
            batch_size = class_preds.shape[1]
            num_classes = all_logits[0].shape[1]

            voted = []
            for i in range(batch_size):
                votes = class_preds[:, i]
                # Count votes for each class
                counts = torch.bincount(votes, minlength=num_classes)
                voted.append(counts)

            # Return as "logits" (vote counts)
            return torch.stack(voted).float()

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def run_inference(self, model, dataloader, dataset_size, device, class_names=None, task_type="classification", num_classes=None):
        """
        Run ensemble inference on the dataset.

        Note: The 'model' parameter is ignored for ensemble inference since we
        load our own models from checkpoints.

        Args:
            model: Ignored (ensemble loads its own models)
            dataloader: DataLoader for test/inference data
            dataset_size: Total size of the test dataset
            device: Device to run inference on
            class_names: List of class names for human-readable results (optional)

        Returns:
            Tuple of (test_acc, per_sample_results) where:
                - test_acc: Overall accuracy as a tensor
                - per_sample_results: List of (true_label, pred_label, is_correct) tuples
        """
        running_corrects = 0
        per_sample_results = []

        total_batches = len(dataloader)
        logger.info(f"Running ensemble inference on {dataset_size} samples...")

        # Iterate over data
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Collect predictions from all models
            all_logits = []

            with torch.no_grad():
                for model_instance in self.models:
                    outputs = model_instance(inputs)
                    all_logits.append(outputs)

            # Aggregate predictions
            aggregated_logits = self._aggregate_logits(all_logits)
            _, preds = torch.max(aggregated_logits, 1)

            # Store per-sample results
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                is_correct = pred_label == true_label

                if class_names:
                    true_name = class_names[true_label]
                    pred_name = class_names[pred_label]
                    per_sample_results.append((true_name, pred_name, is_correct))
                else:
                    per_sample_results.append((true_label, pred_label, is_correct))

            # Statistics
            running_corrects += torch.sum(preds == labels.data)

            # Progress logging
            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                logger.info(
                    f"Progress: {batch_idx + 1}/{total_batches} batches "
                    f"({(batch_idx + 1) / total_batches * 100:.1f}%)"
                )

        test_acc = running_corrects.double() / dataset_size

        logger.info(f"Overall Test Acc (Ensemble): {test_acc:.4f}")

        return test_acc, per_sample_results

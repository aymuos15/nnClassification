"""Combined TTA + Ensemble inference strategy for maximum performance."""

import os

import torch
from loguru import logger

from ml_src.core.config import load_config
from ml_src.core.inference.base import BaseInferenceStrategy
from ml_src.core.network import get_model, load_model
from ml_src.core.run import get_run_dir_from_checkpoint
from ml_src.core.transforms.tta import aggregate_predictions, get_tta_transforms


class TTAEnsembleInference(BaseInferenceStrategy):
    """
    Combined TTA + Ensemble inference strategy.

    This strategy combines both Test-Time Augmentation and model ensembling
    for maximum prediction robustness. It:
    1. Loads multiple models from different checkpoints
    2. Applies TTA to each test image
    3. Runs all models on all augmented versions
    4. Aggregates predictions across both TTA and ensemble

    This is the most powerful but also slowest inference strategy.

    Example:
        >>> # Config with TTA + Ensemble
        >>> config = {
        ...     'inference': {
        ...         'strategy': 'tta_ensemble',
        ...         'tta': {
        ...             'augmentations': ['horizontal_flip', 'vertical_flip'],
        ...             'aggregation': 'mean'
        ...         },
        ...         'ensemble': {
        ...             'checkpoints': [
        ...                 'runs/fold_0/weights/best.pt',
        ...                 'runs/fold_1/weights/best.pt'
        ...             ],
        ...             'aggregation': 'soft_voting'
        ...         }
        ...     }
        ... }
        >>> strategy = TTAEnsembleInference(
        ...     checkpoints=['runs/fold_0/weights/best.pt', ...],
        ...     config=config,
        ...     device='cuda:0',
        ...     tta_augmentations=['horizontal_flip'],
        ...     tta_aggregation='mean',
        ...     ensemble_aggregation='soft_voting'
        ... )
    """

    def __init__(
        self,
        checkpoints,
        config,
        device,
        tta_augmentations=None,
        tta_aggregation="mean",
        ensemble_aggregation="soft_voting",
        ensemble_weights=None,
        use_ema: bool = False,
    ):
        """
        Initialize TTA + Ensemble inference strategy.

        Args:
            checkpoints: List of checkpoint paths to load
            config: Configuration dictionary (needed to recreate models)
            device: Device to run inference on
            tta_augmentations: List of TTA augmentation names
                If None, defaults to ['horizontal_flip']
            tta_aggregation: TTA aggregation method - 'mean', 'max', or 'voting'
            ensemble_aggregation: Ensemble aggregation method
                - 'soft_voting': Average logits (recommended)
                - 'hard_voting': Majority vote on predicted classes
                - 'weighted': Weighted average using ensemble_weights
            ensemble_weights: List of weights for each model (only for 'weighted')
        """
        if not checkpoints or len(checkpoints) == 0:
            raise ValueError("Must provide at least one checkpoint for ensemble")

        if tta_augmentations is None:
            tta_augmentations = ["horizontal_flip"]

        self.checkpoints = checkpoints
        self.config = config
        self.device = device
        self.tta_augmentations = tta_augmentations
        self.tta_aggregation = tta_aggregation
        self.ensemble_aggregation = ensemble_aggregation
        self.ensemble_weights = ensemble_weights
        self.use_ema = use_ema

        # Create TTA transform
        self.tta_transform = get_tta_transforms(tta_augmentations)

        # Validate ensemble aggregation
        valid_methods = {"soft_voting", "hard_voting", "weighted"}
        if ensemble_aggregation not in valid_methods:
            raise ValueError(
                f"Unknown ensemble aggregation: '{ensemble_aggregation}'. "
                f"Valid options: {sorted(valid_methods)}"
            )

        # Setup weights for weighted ensemble
        if ensemble_aggregation == "weighted":
            if ensemble_weights is None:
                self.ensemble_weights = [1.0 / len(checkpoints)] * len(checkpoints)
                logger.info("No weights provided, using equal weights")
            elif len(ensemble_weights) != len(checkpoints):
                raise ValueError(
                    f"Number of weights ({len(ensemble_weights)}) must match "
                    f"number of checkpoints ({len(checkpoints)})"
                )
            else:
                # Normalize weights
                weight_sum = sum(ensemble_weights)
                self.ensemble_weights = [w / weight_sum for w in ensemble_weights]

        logger.info(f"TTA + Ensemble inference:")
        logger.info(f"  - Models: {len(checkpoints)}")
        logger.info(f"  - TTA augmentations: {tta_augmentations}")
        logger.info(f"  - TTA aggregation: {tta_aggregation}")
        logger.info(f"  - Ensemble aggregation: {ensemble_aggregation}")
        if ensemble_aggregation == "weighted":
            logger.info(f"  - Ensemble weights: {self.ensemble_weights}")

        # Load all models
        self.models = self._load_models()

    def _load_models(self):
        """Load all models from checkpoints."""
        models = []

        for i, checkpoint_path in enumerate(self.checkpoints):
            logger.info(f"Loading model {i + 1}/{len(self.checkpoints)}: {checkpoint_path}")

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            self._validate_checkpoint_config(checkpoint_path)

            model = get_model(self.config, self.device)
            model = load_model(model, checkpoint_path, self.device, use_ema=self.use_ema)
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

    def _aggregate_ensemble_logits(self, all_logits):
        """
        Aggregate logits from multiple models.

        Args:
            all_logits: List of logit tensors from each model

        Returns:
            Aggregated logits
        """
        if self.ensemble_aggregation == "soft_voting":
            return torch.stack(all_logits).mean(dim=0)

        elif self.ensemble_aggregation == "weighted":
            weighted = torch.stack(
                [logits * weight for logits, weight in zip(all_logits, self.ensemble_weights)]
            )
            return weighted.sum(dim=0)

        elif self.ensemble_aggregation == "hard_voting":
            class_preds = torch.stack([logits.argmax(dim=1) for logits in all_logits])
            batch_size = class_preds.shape[1]
            num_classes = all_logits[0].shape[1]

            voted = []
            for i in range(batch_size):
                votes = class_preds[:, i]
                counts = torch.bincount(votes, minlength=num_classes)
                voted.append(counts)

            return torch.stack(voted).float()

        else:
            raise ValueError(f"Unknown aggregation method: {self.ensemble_aggregation}")

    def run_inference(self, model, dataloader, dataset_size, device, class_names=None):
        """
        Run TTA + Ensemble inference on the dataset.

        For each image:
        1. Create multiple augmented versions (TTA)
        2. Run all models on all augmented versions
        3. Aggregate TTA predictions for each model
        4. Aggregate ensemble predictions across models

        Args:
            model: Ignored (we load our own models)
            dataloader: DataLoader for test/inference data
            dataset_size: Total size of the test dataset
            device: Device to run inference on
            class_names: List of class names for human-readable results (optional)

        Returns:
            Tuple of (test_acc, per_sample_results)
        """
        running_corrects = 0
        per_sample_results = []

        total_batches = len(dataloader)
        logger.info(f"Running TTA+Ensemble inference on {dataset_size} samples...")
        logger.warning(
            "This is the most powerful but slowest inference method. "
            "Consider using just TTA or Ensemble for faster inference."
        )

        # Iterate over data
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Apply TTA augmentations to the entire batch
            augmented_batches = self.tta_transform.apply_batch(inputs)

            # Collect predictions from all models on augmented batches
            model_predictions = []

            with torch.no_grad():
                for model_instance in self.models:
                    aug_outputs = []
                    for aug_batch in augmented_batches:
                        aug_batch = aug_batch.to(device)
                        outputs = model_instance(aug_batch)
                        aug_outputs.append(outputs)

                    # Aggregate TTA predictions for this model
                    tta_aggregated = aggregate_predictions(
                        aug_outputs, method=self.tta_aggregation
                    )
                    model_predictions.append(tta_aggregated)

            # Aggregate ensemble predictions
            batch_outputs = self._aggregate_ensemble_logits(model_predictions)
            _, preds = torch.max(batch_outputs, 1)

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

        logger.info(f"Overall Test Acc (TTA+Ensemble): {test_acc:.4f}")

        return test_acc, per_sample_results

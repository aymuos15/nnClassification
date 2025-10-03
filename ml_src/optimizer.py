"""Optimizer and learning rate scheduler module."""

import torch.optim as optim
from torch.optim import lr_scheduler


def get_optimizer(model, config):
    """
    Create an optimizer based on the configuration.

    Args:
        model: The model to optimize
        config: Configuration dictionary with optimizer settings

    Returns:
        Configured optimizer (default: SGD with momentum)
    """
    lr = config['optimizer']['lr']
    momentum = config['optimizer'].get('momentum', 0.9)

    # Currently uses SGD with momentum
    # In the future, can be extended to support multiple optimizers:
    # - Adam
    # - AdamW
    # - RMSprop
    # - Lion
    #
    # Example future usage:
    # optimizer_type = config['optimizer'].get('type', 'sgd')
    # if optimizer_type == 'adam':
    #     return optim.Adam(model.parameters(), lr=lr)
    # elif optimizer_type == 'adamw':
    #     weight_decay = config['optimizer'].get('weight_decay', 0.01)
    #     return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    return optimizer


def get_scheduler(optimizer, config):
    """
    Create a learning rate scheduler based on the configuration.

    Args:
        optimizer: The optimizer to schedule
        config: Configuration dictionary with scheduler settings

    Returns:
        Configured learning rate scheduler (default: StepLR)
    """
    step_size = config['scheduler'].get('step_size', 7)
    gamma = config['scheduler'].get('gamma', 0.1)

    # Currently uses StepLR
    # In the future, can be extended to support multiple schedulers:
    # - CosineAnnealingLR
    # - ReduceLROnPlateau
    # - OneCycleLR
    # - ExponentialLR
    #
    # Example future usage:
    # scheduler_type = config['scheduler'].get('type', 'step')
    # if scheduler_type == 'cosine':
    #     T_max = config['training']['num_epochs']
    #     return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    # elif scheduler_type == 'reduce_on_plateau':
    #     return lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return scheduler

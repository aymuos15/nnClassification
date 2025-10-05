"""Configuration override utilities."""


def override_config(config, args):
    """
    Override configuration with command-line arguments.

    Args:
        config: Configuration dictionary
        args: Parsed command-line arguments (from argparse)

    Returns:
        tuple: (updated_config, overrides_list)
            - updated_config: Configuration with overrides applied
            - overrides_list: List of override strings for run naming
    """
    overrides = []

    if hasattr(args, "dataset_name") and args.dataset_name:
        config["data"]["dataset_name"] = args.dataset_name
    if hasattr(args, "data_dir") and args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if hasattr(args, "batch_size") and args.batch_size:
        config["training"]["batch_size"] = args.batch_size
        overrides.append(f"batch_{args.batch_size}")
    if hasattr(args, "num_workers") and args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if hasattr(args, "num_epochs") and args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
        overrides.append(f"epochs_{args.num_epochs}")
    if hasattr(args, "lr") and args.lr:
        config["optimizer"]["lr"] = args.lr
        overrides.append(f"lr_{args.lr}")
    if hasattr(args, "momentum") and args.momentum:
        config["optimizer"]["momentum"] = args.momentum
    if hasattr(args, "step_size") and args.step_size:
        config["scheduler"]["step_size"] = args.step_size
    if hasattr(args, "gamma") and args.gamma:
        config["scheduler"]["gamma"] = args.gamma
    if hasattr(args, "fold") and args.fold is not None:
        config["data"]["fold"] = args.fold
        overrides.append(f"fold_{args.fold}")

    return config, overrides

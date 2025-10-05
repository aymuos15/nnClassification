"""Logging setup utilities."""

import os

from loguru import logger


def setup_console_logger(level="INFO"):
    """
    Setup console logger with color formatting.

    Args:
        level: Logging level (default: INFO)
    """
    logger.add(
        lambda msg: print(msg, end=""),
        format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | <level>{level: <8}</level> | {message}",
        colorize=True,
        level=level,
    )


def setup_file_logger(run_dir, filename="train.log", level="DEBUG"):
    """
    Setup file logger without color formatting.

    Args:
        run_dir: Run directory path
        filename: Log filename (default: train.log)
        level: Logging level (default: DEBUG)

    Returns:
        str: Path to log file
    """
    log_path = os.path.join(run_dir, "logs", filename)
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="10 MB",
        retention="30 days",
        level=level,
    )
    return log_path


def setup_logging(run_dir=None, filename="train.log", console_level="INFO", file_level="DEBUG"):
    """
    Setup logging to both console and file (if run_dir provided).

    Args:
        run_dir: Run directory path (optional, if None only console logging is setup)
        filename: Log filename (default: train.log)
        console_level: Console logging level (default: INFO)
        file_level: File logging level (default: DEBUG)

    Returns:
        str or None: Path to log file if run_dir provided, None otherwise
    """
    # Remove default handler
    logger.remove()

    # Add console logger
    setup_console_logger(level=console_level)

    # Add file logger if run_dir provided
    if run_dir:
        log_path = setup_file_logger(run_dir, filename, level=file_level)
        logger.info(f"Logging to {log_path}")
        return log_path

    return None

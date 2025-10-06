"""Search visualization utilities for Optuna studies."""

import os

from loguru import logger


def visualize_search_study(
    study_name: str,
    storage: str = "sqlite:///optuna_studies.db",
    plot_type: str = "all",
    params: list = None,
    output_dir: str = None,
):
    """
    Generate visualizations for an Optuna study.

    Args:
        study_name: Name of the study to visualize
        storage: Storage URL (e.g., 'sqlite:///optuna_studies.db')
        plot_type: Type of plot to generate ('all', 'optimization_history',
                   'param_importances', 'slice', 'contour', 'parallel_coordinate',
                   'intermediate_values')
        params: List of parameter names for contour/slice plots
        output_dir: Directory to save plots (default: study directory)

    Example:
        >>> visualize_search_study('my_study', plot_type='optimization_history')
        >>> visualize_search_study('my_study', plot_type='contour', params=['lr', 'batch_size'])
    """
    try:
        import optuna
        from optuna.visualization import (
            plot_contour,
            plot_intermediate_values,
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )
    except ImportError:
        logger.error("Optuna is not installed. Install with:\n  pip install -e '.[optuna]'")
        return

    # Load study
    try:
        logger.info(f"Loading study: {study_name}")
        study = optuna.load_study(study_name=study_name, storage=storage)
        logger.success(f"Study loaded: {len(study.trials)} trials")
    except Exception as e:
        logger.error(f"Failed to load study: {e}")
        logger.error(
            f"Make sure the study exists and storage is correct.\n"
            f"Study name: {study_name}\n"
            f"Storage: {storage}"
        )
        return

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join("runs", "optuna_studies", study_name, "visualizations")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving plots to: {output_dir}")

    # Generate plots
    plots_generated = []

    if plot_type in ["all", "optimization_history"]:
        try:
            logger.info("Generating optimization history plot...")
            fig = plot_optimization_history(study)
            path = os.path.join(output_dir, "optimization_history.html")
            fig.write_html(path)
            plots_generated.append(path)
            logger.success("Saved: optimization_history.html")
        except Exception as e:
            logger.warning(f"Failed to generate optimization_history: {e}")

    if plot_type in ["all", "param_importances"]:
        try:
            logger.info("Generating parameter importances plot...")
            fig = plot_param_importances(study)
            path = os.path.join(output_dir, "param_importances.html")
            fig.write_html(path)
            plots_generated.append(path)
            logger.success("Saved: param_importances.html")
        except Exception as e:
            logger.warning(f"Failed to generate param_importances: {e}")

    if plot_type in ["all", "parallel_coordinate"]:
        try:
            logger.info("Generating parallel coordinate plot...")
            fig = plot_parallel_coordinate(study)
            path = os.path.join(output_dir, "parallel_coordinate.html")
            fig.write_html(path)
            plots_generated.append(path)
            logger.success("Saved: parallel_coordinate.html")
        except Exception as e:
            logger.warning(f"Failed to generate parallel_coordinate: {e}")

    if plot_type in ["all", "slice"]:
        try:
            logger.info("Generating slice plot...")
            fig = plot_slice(study, params=params)
            path = os.path.join(output_dir, "slice.html")
            fig.write_html(path)
            plots_generated.append(path)
            logger.success("Saved: slice.html")
        except Exception as e:
            logger.warning(f"Failed to generate slice: {e}")

    if plot_type == "contour":
        if params is None or len(params) < 2:
            logger.warning(
                "Contour plot requires at least 2 parameters. Use --params param1 param2"
            )
        else:
            try:
                logger.info(f"Generating contour plot for {params}...")
                fig = plot_contour(study, params=params[:2])  # Use first 2 params
                param_str = "_".join(params[:2])
                path = os.path.join(output_dir, f"contour_{param_str}.html")
                fig.write_html(path)
                plots_generated.append(path)
                logger.success(f"Saved: contour_{param_str}.html")
            except Exception as e:
                logger.warning(f"Failed to generate contour: {e}")

    if plot_type in ["all", "intermediate_values"]:
        try:
            logger.info("Generating intermediate values plot...")
            fig = plot_intermediate_values(study)
            path = os.path.join(output_dir, "intermediate_values.html")
            fig.write_html(path)
            plots_generated.append(path)
            logger.success("Saved: intermediate_values.html")
        except Exception as e:
            logger.warning(f"Failed to generate intermediate_values: {e}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Visualization Summary")
    logger.info("=" * 70)
    logger.info(f"Study: {study_name}")
    logger.info(f"Trials: {len(study.trials)}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Plots generated: {len(plots_generated)}")

    if plots_generated:
        logger.info("\nGenerated plots:")
        for plot_path in plots_generated:
            logger.info(f"  - {plot_path}")
        logger.info(f"\nOpen plots in browser: file://{os.path.abspath(output_dir)}/")
    else:
        logger.warning("No plots were generated")

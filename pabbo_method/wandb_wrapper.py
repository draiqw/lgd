"""
Weights & Biases (wandb) wrapper for PABBO training.

This module provides a thin wrapper around wandb to handle:
- Optional wandb usage (can be disabled)
- Configuration management
- Artifact saving
"""

import os
from typing import Optional, Any, Dict
from omegaconf import DictConfig, OmegaConf


def init(
    config: Optional[DictConfig] = None,
    project: str = "PABBO",
    name: Optional[str] = None,
    group: Optional[str] = None,
    job_type: str = "train",
    tags: Optional[list] = None,
    dir: Optional[str] = None,
    **kwargs
) -> Optional[Any]:
    """
    Initialize Weights & Biases run.

    This function attempts to import and initialize wandb. If wandb is not installed
    or initialization fails, it returns None and prints a warning.

    Args:
        config: Optional configuration object (DictConfig or dict)
        project: wandb project name
        name: Run name (displayed in wandb UI)
        group: Run group for organizing experiments
        job_type: Type of job (train, eval, etc.)
        tags: List of tags for the run
        dir: Directory to save wandb files
        **kwargs: Additional arguments passed to wandb.init()

    Returns:
        wandb.Run object if successful, None otherwise
    """
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed. Training will continue without wandb logging.")
        print("To enable wandb logging, install it with: pip install wandb")
        return None

    try:
        # Convert DictConfig to dict for wandb
        config_dict = None
        if config is not None:
            if isinstance(config, DictConfig):
                config_dict = OmegaConf.to_container(config, resolve=True)
            else:
                config_dict = config

        # Initialize wandb
        run = wandb.init(
            project=project,
            name=name,
            group=group,
            job_type=job_type,
            tags=tags,
            config=config_dict,
            dir=dir,
            **kwargs
        )

        print(f"wandb initialized: {run.name} (id: {run.id})")
        print(f"View run at: {run.url}")

        return run

    except Exception as e:
        print(f"WARNING: Failed to initialize wandb: {e}")
        print("Training will continue without wandb logging.")
        return None


def save_artifact(
    run: Optional[Any],
    local_path: str,
    name: str = "checkpoint",
    type: str = "model",
    aliases: Optional[list] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save a file or directory as a wandb artifact.

    This function creates and logs an artifact to wandb. If wandb is not initialized
    or the run is None, it silently returns without error.

    Args:
        run: wandb.Run object (from wandb.init() or wandb_wrapper.init())
        local_path: Path to the local file or directory to save
        name: Name of the artifact
        type: Type of artifact (model, dataset, etc.)
        aliases: List of aliases for the artifact (e.g., ['latest', 'best'])
        metadata: Optional metadata dict to attach to artifact

    Returns:
        None
    """
    # If run is None or wandb not available, silently return
    if run is None:
        return

    try:
        import wandb
    except ImportError:
        return

    try:
        # Check if file/directory exists
        if not os.path.exists(local_path):
            print(f"WARNING: Path {local_path} does not exist. Skipping artifact save.")
            return

        # Create artifact
        artifact = wandb.Artifact(
            name=name,
            type=type,
            metadata=metadata or {}
        )

        # Add file or directory
        if os.path.isfile(local_path):
            artifact.add_file(local_path)
        elif os.path.isdir(local_path):
            artifact.add_dir(local_path)
        else:
            print(f"WARNING: {local_path} is neither a file nor directory. Skipping.")
            return

        # Log artifact with aliases
        run.log_artifact(artifact, aliases=aliases or ['latest'])

        print(f"Saved artifact '{name}' from {local_path}")

    except Exception as e:
        print(f"WARNING: Failed to save artifact: {e}")
        # Continue without crashing


def log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True
) -> None:
    """
    Log metrics to wandb.

    This is a convenience function that wraps wandb.log(). If wandb is not
    initialized, it silently returns.

    Args:
        data: Dictionary of metric name -> value
        step: Optional step number
        commit: Whether to commit this log (default True)

    Returns:
        None
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(data, step=step, commit=commit)
    except (ImportError, AttributeError):
        pass


def finish() -> None:
    """
    Finish the current wandb run.

    This should be called at the end of training/evaluation to properly
    close the wandb run.

    Returns:
        None
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            print("wandb run finished.")
    except (ImportError, AttributeError):
        pass

import logging
import os
import sys
from pathlib import Path
from typing import List

import hydra
import lightning as L
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.train import instantiate_callbacks, instantiate_loggers, test, train
from src.utils.logging_utils import setup_logger

log = logging.getLogger(__name__)


def get_latest_optimization_results(
    base_path: Path, file_name: str = "optimization_results.yaml"
):
    return max(base_path.iterdir(), key=lambda d: d.stat().st_mtime) / file_name


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "train_optimized_log.log")

    # Load the optimization results
    base_path = Path(cfg.paths.log_dir) / "train" / "multiruns"
    optimization_results_path = get_latest_optimization_results(base_path)
    if not optimization_results_path.exists():
        raise FileNotFoundError(
            f"Optimization results file not found at {optimization_results_path}"
        )

    best_params = OmegaConf.load(optimization_results_path)

    # Update the configuration with the best parameters
    for param, value in best_params["best_params"].items():
        OmegaConf.update(cfg, param, value, merge=True)

    # Print the final configuration
    log.info("Final configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    if cfg.get("train"):
        train(cfg, trainer, model, datamodule)

    # Test the model
    if cfg.get("test"):
        test(cfg, trainer, model, datamodule)


if __name__ == "__main__":
    main()
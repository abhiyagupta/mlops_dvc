import os
from pathlib import Path
import logging
import lightning as L
import hydra
from omegaconf import DictConfig,OmegaConf
# import lightning as Ldata
from lightning.pytorch.loggers import Logger
from typing import List

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# # Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper
# Imports after setting up the root
# from src.utils.logging_utils import (
#     setup_logger,
#     task_wrapper,
#     logger,
#     log_metrics_table,
# )




log = logging.getLogger(__name__)


def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callback_cfg.items():
        #if "_target_" in cb_conf:
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


@task_wrapper
def train(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting training!")
    # trainer.fit(model, datamodule)
    # Use the train_dataloader() method of the datamodule
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")


@task_wrapper
def test(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting testing!")
    if trainer.checkpoint_callback.best_model_path:
        log.info(
            f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}"
        )
        test_metrics = trainer.test(
            model, dataloaders=datamodule.test_dataloader(), ckpt_path=trainer.checkpoint_callback.best_model_path
        )
    else:
        log.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, dataloaders=datamodule.test_dataloader(),verbose=False)
    log.info(f"Test metrics:\n{test_metrics}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):

    # print the current working directory and content of config
    print(f"Current working directory: {os.getcwd()}")
    print(OmegaConf.to_yaml(cfg=cfg))

     # Resolve the paths
    # cfg.paths.data_dir = os.path.abspath(cfg.paths.data_dir)
    # cfg.paths.log_dir = os.path.abspath(cfg.paths.log_dir)
    # cfg.paths.output_dir = os.path.abspath(cfg.paths.output_dir)



    ## ADDED changes

    # Print resolved paths
    # print(f"Data directory: {cfg.paths.data_dir}")
    # print(f"Log directory: {cfg.paths.log_dir}")
    # print(f"Output directory: {cfg.paths.output_dir}")

    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the log directory exists

    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Log the resolved paths
    # log.info(f"Data directory: {cfg.paths.data_dir}")
    # log.info(f"Log directory: {cfg.paths.log_dir}")
    # log.info(f"Output directory: {cfg.paths.output_dir}")




    ## changes over here



    # # Set up paths
    # log_dir = Path(cfg.paths.log_dir)

    # # Set up logger
    # setup_logger(log_dir / "train_log.log")

   
    try:
        # Initialize DataModule
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    except Exception as e:
        log.exception("Error instantiating datamodule")
        raise  

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

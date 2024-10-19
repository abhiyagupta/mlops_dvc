import pytest
import hydra
from pathlib import Path
import os
import shutil
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from  src.data_modules.dogs_datamodule import DogsBreedDataModule
# Import train function
from src.train import train

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=finetune"],
        )
        return cfg

# Rename this function to start with "test_"
def test_dogs_breed_training(config, tmp_path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")

    # Instantiate components
    datamodule = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer)

    # Run training
    train(config, trainer, model, datamodule)
    
    # Print directory contents for debugging
    print(f"Contents of {tmp_path}:")
    for item in os.listdir(tmp_path):
        print(f"- {item}")
    
    # Check if checkpoints directory exists
    checkpoints_dir = tmp_path / "checkpoints"
    config.trainer.default_root_dir = str(checkpoints_dir)

     # Ensure the checkpoints directory exists
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate components
    datamodule = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer)

    # Run training
    train(config, trainer, model, datamodule)
    
    # Print directory contents for debugging
    print(f"Contents of {tmp_path}:")
    for item in os.listdir(tmp_path):
        print(f"- {item}")

    
    assert checkpoints_dir.exists(), f"Checkpoints directory should be created at {checkpoints_dir}"
    
    # If checkpoints directory exists, check its contents
    if checkpoints_dir.exists():
        print(f"Contents of {checkpoints_dir}:")
        for item in os.listdir(checkpoints_dir):
            print(f"- {item}")
    
    # Add some assertions to verify the training occurred
    assert any(checkpoints_dir.iterdir()), f"At least one checkpoint should be saved in {checkpoints_dir}"

    # Clean up temporary directory after test
    shutil.rmtree(tmp_path) 

# Add a simple test to ensure pytest is running
def test_pytest_is_working():
    assert True, "This test should always pass"
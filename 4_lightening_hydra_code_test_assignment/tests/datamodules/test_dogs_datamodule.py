import pytest
import torch 
import rootutils
import os

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_modules.dogs_datamodule import DogsBreedDataModule

@pytest.fixture
def datamodule():
    print("Creating datamodule fixture")
    print(f"Current working directory: {os.getcwd()}")

    # Use absolute path or path relative to project root
    # data_dir = "/code/data/dogs_dataset"

    data_dir="./data/dogs_dataset"
    
    dm = DogsBreedDataModule(batch_size=8,
                num_workers=0,
                pin_memory=False,
                data_dir= data_dir) #"./data/dogs_dataset")
    print(f"DataModule: {dm}")
    
    print(f"Data directory used: {data_dir}")
    return dm

def test_dogsbreed_data_setup(datamodule):
    print("Running test_dogsbreed_data_setup")
    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.transforms is not None, "Transforms should not be None"
    assert datamodule.train_dataset is not None, "Train dataset should not be None"
    assert datamodule.test_dataset is not None, "Test dataset should not be None"
    assert datamodule.val_dataset is not None, "Validation dataset should not be None"

def test_prepare_data(datamodule):
    print("Running test_prepare_data")
    datamodule.prepare_data()
    datamodule.setup()
    assert len(datamodule.train_dataset) > len(datamodule.test_dataset), "Train dataset should be larger than test dataset"
    assert len(datamodule.train_dataset) > len(datamodule.val_dataset), "Train dataset should be larger than validation dataset"

def test_dogsbreed_dataloaders(datamodule):
    print("Running test_dogsbreed_dataloaders")
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader is not None, "Train dataloader should not be None"
    assert val_loader is not None, "Validation dataloader should not be None"
    assert test_loader is not None, "Test dataloader should not be None"



# #add a simple test that doesn't rely on the datamodule
# def test_simple():
#     print("Running test_simple")
#     assert True
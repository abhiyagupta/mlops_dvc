import pytest
import torch
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.models.dogs_classifier import DogsBreedClassifier



def test_dogsbread_classifier_forward():
    model = DogsBreedClassifier(model_name='resnet18', 
            num_classes=10, 
            pretrained=True,
            trainable=False,
            lr=0.001, 
            weight_decay=.9,
            scheduler_factor=1,
            scheduler_patience=1,
            min_lr=0.1)
    print(model)
    batch_size, channels, height, width = 4, 3, 224, 224
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    assert output.shape == (batch_size, 10)


if __name__=="__main__":
    test_dogsbread_classifier_forward()
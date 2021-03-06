from torchvision.datasets import ImageFolder
import torch

class VGGFace2Dataset(ImageFolder):
    def __init__(self, data_dir, transform):
        super().__init__(data_dir,transform)

    def __getitem__(self, index):
        Image, label = super().__getitem__(index)
        return (Image, label, index)
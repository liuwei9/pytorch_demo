import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)
data_train = datasets.MNIST(root="data/", transform=transform, train=True, download=True)
data_test = datasets.MNIST(root="data/", transform=transform, train=False,download=True)


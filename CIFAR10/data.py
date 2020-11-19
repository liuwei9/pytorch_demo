import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch import nn
from torch import utils
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy
cifar_train = datasets.CIFAR10(root='data/', train=True, transform=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip()
    ]
), download=True)

cifar_test = datasets.CIFAR10(root='data/', train=False, transform=transforms.ToTensor(), download=True)

data_train = DataLoader(cifar_train,batch_size=32,shuffle=True)
data_test = DataLoader(cifar_test,batch_size=32)

if __name__ == '__main__':
    data,label = iter(data_train).__next__()
    print(data.shape)
    print(label.shape)
    img = torchvision.transforms.ToPILImage()(data[0])
    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




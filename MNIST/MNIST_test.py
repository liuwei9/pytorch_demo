import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, ], std=[0.5, ])]
)
data_train = datasets.MNIST(root="data/", transform=transform, train=True, download=True)
data_test = datasets.MNIST(root="data/", transform=transform, train=False, download=True)

data_train_load = torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size=64,
                                              shuffle=True)
data_test_load = torch.utils.data.DataLoader(dataset=data_test,
                                             batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.linear1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 50),
            nn.ReLU(),
            nn.Linear(50, 10),

        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.linear1(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
cost = nn.CrossEntropyLoss().to(device)
optim = torch.optim.Adam(model.parameters())


def train(epoch):
    sum_loss = 0
    for i, data in enumerate(data_train_load):
        inputs, lable = data
        inputs = inputs.to(device)
        lable = lable.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = cost(outputs, lable)
        loss.backward()
        optim.step()
        sum_loss += loss.item()
        if i % 100 == 0:
            print("epoch:" + str(epoch) + "  " + "step" + str(i) + " loss:" + str(sum_loss / 100))
            sum_loss = 0


def train(epoch):
    sum_loss = 0
    model.train()
    for batch, data in enumerate(data_train_load):
        inputs, lable = data
        inputs = inputs.to(device)
        lable = lable.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = cost(outputs, lable)
        loss.backward()
        optim.step()
        sum_loss += loss.item()
        if batch % 100 == 0:
            print("epoch: " + str(epoch) + "  " + "batch: " + str(batch) + " loss: " + str(sum_loss / 100))
            torch.save(model,"logs/"+"epoch" + str(epoch) + "  " + "batch" + str(batch) + " loss" + str(sum_loss / 100)+".pth")
            sum_loss = 0


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_test_load:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += cost(output, target).item()
            pred = output.max(1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_test_load.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_test_load.dataset),
        100. * correct / len(data_test_load.dataset)))


for i in range(1, 21):
    train(i)
    test(i)

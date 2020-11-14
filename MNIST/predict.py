import torch
import cv2 as cv
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
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
model = torch.load("logs/epoch20  batch900 loss0.0020713851519485616.pth")
for i in range(1,6):
    img = cv.imread(str(i)+".jpg")
    cv.imshow(str(i),img)
    img = cv.resize(img,(28,28))

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img,0)
    img = np.expand_dims(img,0)
    img=torch.from_numpy(img)
    img = img.to(device)
    outputs = model(img)
    prob = F.softmax(outputs, dim=1)
    #prob = prob.cpu().numpy()  #用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    print(prob)  #prob是10个分类的概率
    pred = torch.argmax(prob) #选出概率最大的一个
    print(pred.item())

cv.waitKey(0)
cv.destroyAllWindows()
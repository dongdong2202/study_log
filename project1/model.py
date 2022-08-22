from sklearn.feature_selection import SelectFpr
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3) # 输入通道1便成输出6通道 kerl=3x3 =》（1, ）
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 3) # inChannel = 6 outChannel=16, ker=3x3
        self.pool2  = nn.MaxPool2d(2,2)
        self.fc3 = nn.Linear(16*6*6, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        print(x.shape) #输入（batch, c, 32,32） kerl=3x3 =》（batch, c, 30,30）
        x = self.pool1(torch.relu(self.conv1(x))) # 
        print(x.shape) # 因为pool=2,所以 就变成30/2=15
        x = self.pool2(torch.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


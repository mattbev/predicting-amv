import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,3))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,3))

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(in_features=64*54*24, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class SimpleResNet(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,3))

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), padding=(1,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,3))

        # self.conv1, self.pool1 = self.block()
        # self.conv2, self.pool2 = self.block()
        # self.conv3, self.pool3 = self.block()

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(in_features=3*56*24, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

        self.skip = nn.Identity()

    # @staticmethod
    # def block(in_channels=3, out_channels=3, conv_kernel_size=(3,3), pool_kernel_size=(2,3), padding=(1,1)):
    #     conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel_size, padding=padding)
    #     pool = nn.MaxPool2d(kernel_size=pool_kernel_size)
    #     return conv, pool

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = F.relu(x)
        x = x + self.skip(res)
        x = self.pool1(x)

        res = x

        x = self.conv2(x)
        x = F.relu(x)
        x = x + self.skip(res)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


    
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

def doubleConv(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                     kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                     kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

def singleConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels=out_channels,
         kernel_size=3, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
    )

def fullyConnected(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features = in_features, out_features = out_features),
        nn.ReLU(inplace = True),
    )

class VGGModel(nn.Module):
    def __init__(self,channels, height, weight, classes):
        super(VGGModel, self).__init__()

        # Convolutional Area
        self.dblconv1 = doubleConv(3, 64)

        self.dblconv2 = doubleConv(64, 128)

        self.dblconv3 = doubleConv(128, 256)
        self.singleConv3 = singleConv(256, 256)

        self.dblconv4 = doubleConv(256, 512)
        self.singleConv4 = singleConv(512, 512)

        self.dblconv5 = doubleConv(512, 512)
        self.singleConv5 = singleConv(512, 512)

        self.maxPooling = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        # Fully Connected Area

        self.heightWeight = height / 2**5
        self._dim = int(self.heightWeight ** 2 * 512)

        self.fullConnected1 = fullyConnected(self._dim, 4096)
        self.fullConnected2 = fullyConnected(4096, 4096)
        self.fullConnected3 = fullyConnected(4096, 1000)

        self.out = nn.Linear(1000, classes)
    
    def forward(self, x):

        x = self.dblconv1(x)
        x = self.maxPooling(x)
        x = self.dblconv2(x)
        x = self.maxPooling(x)
        x = self.dblconv3(x)
        x = self.singleConv3(x)
        x = self.maxPooling(x)
        x = self.dblconv4(x)
        x = self.singleConv4(x)
        x = self.maxPooling(x)
        x = self.dblconv5(x)
        x = self.singleConv5(x)
        x = self.maxPooling(x)

        x = self.flatten(x)

        x = self.fullConnected1(x)
        x = self.fullConnected2(x)
        x = self.fullConnected3(x)

        x = self.out(x)

        x = nn.Softmax(dim = 1)(x)
        return x


if __name__ == '__main__':
    dummy = torch.rand((4, 3, 224, 224))
    model = VGGModel(3, 224, 224, 10)
    output = model(dummy)
    print(output.shape)
    print(output)


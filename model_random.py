import torch
import torch.nn as nn

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

class NeuralNet(nn.Module):
    def __init__(self,channels, height, weight, classes):
        super(NeuralNet, self).__init__()

        # Double Convs to get features
        ## Down
        self.down1 = doubleConv(3, 16)
        self.down2 = doubleConv(16, 32)
        self.down3 = doubleConv(32, 64)

        ## Up
        self.up1   = doubleConv(64, 32)
        self.up2   = doubleConv(32, 16)
        self.up3   = doubleConv(16, 3)

        # Pooling of the matrix
        self.pooling = nn.MaxPool2d(2, 2)

        # Increasing dimensions
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Flatten the output
        self.flatten = nn.Flatten()

        # Linear Network to classification
        self._dim = height * weight * channels
        self.linear1 = nn.Linear(in_features = self._dim, out_features = 2 * self._dim)
        self.linear2 = nn.Linear(in_features = 2 * self._dim, out_features = 2 * self._dim)
        self.linear3 = nn.Linear(in_features = 2 * self._dim, out_features = 2 * self._dim)
        self.linear4 = nn.Linear(in_features = 2 * self._dim, out_features = self._dim)
        # Out Layer
        self.Out = nn.Linear(self._dim, classes)

    def forward(self, x):
        # Creating Model Structure
        ## Down Sampling
        x = self.down1(x)
        x = self.pooling(x)
        
        x = self.down2(x)
        x = self.pooling(x)
        
        x = self.down3(x)
        x = self.pooling(x)
        
    
        ## Up Sampling
        x = self.upSample(x)
        x = self.up1(x)
        
        x = self.upSample(x)
        x = self.up2(x)
        
        x = self.upSample(x)
        x = self.up3(x)
        

        ###### Flatten ######
        x = self.flatten(x)
        

        ## Fully Connected
        x = self.linear1(x)
        
        x = self.linear2(x)

        x = self.linear3(x)
        
        x = self.linear4(x)
        
        ## Out Layer
        x = self.Out(x)
        
        x = nn.Softmax(dim = 1)(x)

        return x




if __name__ == '__main__':
    dummy = torch.rand((4, 3, 32, 32))
    model = NeuralNet(3, 32, 32, 10)
    output = model(dummy)
    print('size of output..: ', output.shape)
    print('Out ........: ', output)


import torch.nn as nn
import torchvision


class SimpleHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropblock_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.pred = nn.Conv2d(256, 5 + num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropblock = torchvision.ops.DropBlock2d(p=dropblock_p, block_size=3)
        self.stride = 2
 
    def forward(self, x):
        # x: (B, in_channels, H_in, W_in)
        # H_in, W_in : backbone output size
        
        x = self.relu(self.conv1(x))
        x = self.dropblock(x)
        x = self.relu(self.conv2(x))  # (B, 256, H_in/2, W_in/2)

        # (B, 5+num_classes, H_in/2, W_in/2), raw output
        return self.pred(x)
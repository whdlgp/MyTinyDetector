import torch.nn as nn
import torchvision
from torchvision.models import ConvNeXt_Tiny_Weights
 
 
class ConvNeXtBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        convnext = torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features = convnext.features
        self.stride = 32
        self.out_channels = 768
 
    def forward(self, x):
        # (B, 768, H/32, W/32)
        return self.features(x)
        
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
 
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
 
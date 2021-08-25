import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
        # layer 1
        nn.Conv2d(num_classes, 64, (4*4), 2, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # layer 2
        nn.Conv2d(64, 128, (4*4), 2, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # layer 3
        nn.Conv2d(128 , 256, (4*4), 2, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # layer 4
        nn.Conv2d(256, 512, (4*4), 2, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # layer 5
        nn.Conv2d(512, 1, (4*4), 2, bias=False),
        nn.Upsample(scale_factor=32, mode='bilinear')
        )
    
    def forward(self, input):
        return self.main(input)
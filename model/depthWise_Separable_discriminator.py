import torch
from torch import nn


class DW_Discriminator(nn.Module) :
    def __init__(self, num_classes):
        super(DW_Discriminator, self).__init__()
        self.layer1 = depthwise_separable_conv(num_classes , 64)
        self.layer2 = depthwise_separable_conv(64  , 128) 
        self.layer3 = depthwise_separable_conv(128 , 256)
        self.layer4 = depthwise_separable_conv(256 , 512)
        self.layer5 = depthwise_separable_conv(512 , 1  )

        self.LeakyReLU  = nn.LeakyReLU(0.2, inplace=True)
        self.Upsample   = nn.Upsample(scale_factor=32, mode='bilinear')  

    def forward(self, x):
        x   = self.LeakyReLU(self.layer1(x)) # 1/2
        x   = self.LeakyReLU(self.layer2(x)) # 1/4
        x   = self.LeakyReLU(self.layer3(x)) # 1/8
        x   = self.LeakyReLU(self.layer4(x)) # 1/16
        out = self.Upsample (self.layer5(x)) # 1/32 followed by upsampling with factor x32
        
        return out




class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout): # takes as input numper of input channels , number of kernels per layer , number of output channels  
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin , kernel_size=4, stride = 2 , padding=1, groups=nin) # control thes patial size , return :  1/2 the spatial size H * H
        self.pointwise = nn.Conv2d(nin , nout, kernel_size=1) # control the channels , return : output channels = nout

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

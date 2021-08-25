import torch
from torch import nn

#model based on paper 6 and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()


        self.conv1 = nn.Conv2d(num_classes, 64, (4*4), 2, bias=False)
        self.conv2 = nn.Conv2d(64, 128, (4*4), 2, bias=False)
        self.conv3 = nn.Conv2d(128, 256, (4 * 4), 2, bias=False)
        self.conv4 = nn.Conv2d(256, 512, (4*4), 2, bias=False)
        self.conv5 = nn.Conv2d(512, 1, (4*4), 2, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 64)
        self.drop = nn.Dropout(p=0.25)


    def forward(self, input):
        # layer 1:
        x = self.relu(self.conv1(input))
        # layer 2:
        x = self.relu(self.conv2(x))
        # layer 3:
        x = self.relu(self.conv3(x))
        # layer 4:
        x = self.relu(self.conv4(x))
        # layer 5:
        x = self.conv5(x)
        # layer 6(fully connected layer + dropout instead of upsampling):
        x = self.flatten(x)
        x = self.drop(self.fc(x))
        return x
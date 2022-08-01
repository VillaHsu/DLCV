import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34

class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input



class baselineNet(nn.Module):
    def __init__(self, args):
        super(baselineNet, self).__init__()
        ''' declare layers used in this network '''

        self.class_num = 9
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1

        self.pretrained = resnet18(pretrained=True)      
        self.decoder = self._make_decoder()

        print(self.pretrained)
        print(self.decoder)

    def _make_decoder(self):
        layers = []
        layers.append(nn.ConvTranspose2d(512,256,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(256,128,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(128,64,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(64,32,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(32,16,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(16,9,kernel_size=1,stride=1,padding=0,bias=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        out = self.pretrained.layer4(x)
        out = self.decoder(out)
        return out


class improvedNet(nn.Module):
    def __init__(self, args):
        super(improvedNet, self).__init__()
        ''' declare layers used in this network '''

        self.class_num = 9
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1

        self.pretrained = resnet34(pretrained=True)      
        self.decoder = self._make_decoder()

        print(self.pretrained)
        print(self.decoder)

    def _make_decoder(self):
        layers = []
        layers.append(nn.ConvTranspose2d(512,256,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(256,128,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(128,64,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(64,32,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(32,16,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(16,9,kernel_size=1,stride=1,padding=0,bias=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        out = self.pretrained.layer4(x)
        out = self.decoder(out)
        return out
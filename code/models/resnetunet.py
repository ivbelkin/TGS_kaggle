import torch
import torchvision

from torch import nn
from torch.nn import functional as F


class ConvBn2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), 
                 stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ):
        super().__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x ):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class UNetResNet34(nn.Module):

    def load_pretrain(self, pretrain_file):
        self.resnet.load_state_dict(torch.load(pretrain_file, 
                                               map_location=lambda storage, loc: storage))

    def __init__(self ):
        super().__init__()
        self.resnet = torchvision.models.resnet34()

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4 

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512 + 256, 128, 128)
        self.decoder4 = Decoder(256 + 128, 128, 128)
        self.decoder3 = Decoder(128 + 128, 64, 64)
        self.decoder2 = Decoder(64 + 64, 32, 32)
        self.decoder1 = Decoder(32, 32, 32)

        self.logit = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        x = torch.stack([
            (x - mean[0]) / std[0],
            (x - mean[1]) / std[1],
            (x - mean[2]) / std[2],
        ], 1)

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)
        
        f = self.decoder5(torch.cat([f, e5], 1))
        f = self.decoder4(torch.cat([f, e4], 1))
        f = self.decoder3(torch.cat([f, e3], 1))
        f = self.decoder2(torch.cat([f, e2], 1))
        f = self.decoder1(f)

        f = F.dropout2d(f, p=0.20)
        logit = self.logit(f)
        
        return logit

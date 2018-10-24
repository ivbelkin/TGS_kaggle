import torch

from code.models.common import *


class BasicEncoderBlock(torch.nn.Module):
    """Simple encoder block
    
    Parameters:
        x: pytorch tensor of shape (batch_size, in_channel, height, width)
        
    Returns:
        x: pytorch tensor of shape (batch_size, out_channels, height // 2, width // 2)
        x_trace: pytorch tensor of shape (batch_size, out_channels, height, width)
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = ConvBnRelu(in_channels, out_channels, 3)
        self.conv_2 = ConvBnRelu(out_channels, out_channels, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
    
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        trace = x
        x = self.pool(x)
        return x, trace
    

class BasicDecoderBlock(torch.nn.Module):
    """Simple decoder block
    
    Parameters:
        x: pytorch tensor of shape (batch_size, in_channels, height, width)
        x_trace: pytorch tensor of shape (batch_size, skip_channels, height, width)
    Returns:
        x: pytorch tensor of shape (batch_size, out_channels, 2 * height, 2 * width)
    """
    
    def __init__(self, in_channels, skip_channels, mid_channels, out_channels):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=(2, 2), mode="bilinear")
        self.conv_1 = ConvBnRelu(in_channels + skip_channels, mid_channels, 3)
        self.conv_2 = ConvBnRelu(mid_channels, out_channels, 3)
        
    
    def forward(self, x, x_trace):
        x = self.upsample(x)
        x = torch.cat((x, x_trace), dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class BasicUNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
       
        self.down_1 = BasicEncoderBlock(1, 16)
        self.down_2 = BasicEncoderBlock(16, 32)
        self.down_3 = BasicEncoderBlock(32, 64)
        self.down_4 = BasicEncoderBlock(64, 128)

        self.center = torch.nn.Sequential(
            ConvBnRelu(128, 256, 3),
            torch.nn.Dropout(0.5),
            ConvBnRelu(256, 256, 3),
            torch.nn.Dropout(0.5),
        )
        
        self.up_1 = BasicDecoderBlock(256, 128, 32, 32)
        self.up_2 = BasicDecoderBlock(32, 64, 32, 32)
        self.up_3 = BasicDecoderBlock(32, 32, 32, 32)
        self.up_4 = BasicDecoderBlock(32, 16, 16, 16)
        
        self.conv = torch.nn.Conv2d(in_channels=16, 
                                    out_channels=1, 
                                    kernel_size=(1, 1))
    
    
    def forward(self, x): 
        x, trace_1 = self.down_1(x) 
        x, trace_2 = self.down_2(x)
        x, trace_3 = self.down_3(x)
        x, trace_4 = self.down_4(x)
        
        x = self.center(x)
        
        x = self.up_1(x, trace_4)
        x = self.up_2(x, trace_3)
        x = self.up_3(x, trace_2)
        x = self.up_4(x, trace_1)

        x = self.conv(x)
        
        x = torch.sigmoid(x)
        
        return x

import torch

from code.models.common import *
from code.models.encoders.basic import *
    
    
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
        self.conv_1 = ConvRelu(in_channels + skip_channels, mid_channels, 3)
        self.conv_2 = ConvRelu(mid_channels, out_channels, 3)
        
    
    def forward(self, x, x_trace):
        x = torch.cat((x, x_trace), dim=1)
        x = self.upsample(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    

class BasicUNet(torch.nn.Module):
    
    def __init__(self, encoder, depth):
        super().__init__()
       
        self.encoder = BasicEncoder(1, 8, 4, BasicEncoderBlock)

        self.center = torch.nn.Sequential(
            ConvBnRelu(64, 128, 3),
            torch.nn.Dropout(0.5),
            ConvBnRelu(128, 128, 3),
            torch.nn.Dropout(0.5),
        )
        
        self.up_1 = DecoderBlock(128, 64, 64)
        self.up_2 = DecoderBlock(64, 32, 32)
        self.up_3 = DecoderBlock(32, 16, 16)
        self.up_4 = DecoderBlock(16, 8, 8)
        
        self.conv = torch.nn.Conv2d(in_channels=8, 
                                    out_channels=1, 
                                    kernel_size=(1, 1))
    
    def forward(self, x):      
        trace_1, trace_2, trace_3, trace_4 = self.encoder(x)
        
        x = self.center(trace_4)
        
        x = self.up_1(x, x_trace_4)
        x = self.up_2(x, x_trace_3)
        x = self.up_3(x, x_trace_2)
        x = self.up_4(x, x_trace_1)
        
        x = self.conv(x)
        
        x = torch.sigmoid(x)
        
        return x

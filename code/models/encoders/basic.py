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
    
    
class BasicEncoder(torch.nn.Module):
    """Simple encoder with autoconfig for depth and init number of channels
     
    Just apply Block's sequentually and return their traces.
    
    Parameters:
        x: pytorch tensor of shape (batch_size)
    """
    
    def __init__(self, in_channels, init_channels, depth, Block):
        super().__init__()
        self.depth = depth
        self.blocks = []
        
        block = Block(in_channels, init_channels)
        self.blocks.append(block)
        self.add_module("layer{}".format(1), block)
        
        for i in range(depth - 1):
            block = Block(init_channels * 2 ** i, 2 * init_channels * 2 ** i)
            self.blocks.append(block)
            self.add_module("layer{}".format(i + 2), block)
    
    def forward(self, x):
        traces = []
        for i in range(self.depth):
            x, trace = self.blocks[i](x)
            traces.append(trace)
        return traces

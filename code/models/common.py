import torch
                
                
class ConvRelu(torch.nn.Module):
    """Basic block of convolution followed by relu activation.
    
    Convolution uses square filter and 'same' padding.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = torch.nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=(kernel_size, kernel_size), 
                                    padding=(padding, padding))
        self.relu = torch.nn.ReLU()
    
    def forward(self, x, **kwargs):
        x = self.conv(x)
        x = self.relu(x)
        return x

    
class ConvBnRelu(torch.nn.Module):
    """Basic block of convolution followed by batch normalization and relu activation.
    
    Convolution uses square filter and 'same' padding.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = torch.nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=(kernel_size, kernel_size), 
                                    padding=(padding, padding))
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

import torch

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock


class ResNetEncoder(resnet.ResNet):
    """Resnet-based encoder
    
    forward method returns a tuple of four feature maps extracted from
    different depths.
    """
    
    def __init__(self, block, layers, path_to_weights):
        super().__init__(block, layers)
        del self.avgpool, self.fc
        self.load_state_dict(torch.load(path_to_weights), strict=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_trace1 = self.layer1(x)
        x_trace2 = self.layer2(x_trace1)
        x_trace3 = self.layer3(x_trace2)
        x = self.layer4(x_trace3)

        return x_trace1, x_trace2, x_trace3, x
    
    
class ResNet18Encoder(ResNetEncoder):
    
    def __init__(self, path_to_weights):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2], path_to_weights)
        

class ResNet34Encoder(ResNetEncoder):
    
    def __init__(self, path_to_weights):
        super().__init__(resnet.BasicBlock, [3, 4, 6, 3], path_to_weights)

        
class ResNet50Encoder(ResNetEncoder):
    
    def __init__(self, path_to_weights):
        super().__init__(resnet.Bottleneck, [3, 4, 6, 3], path_to_weights)
        
        
class ResNet101Encoder(ResNetEncoder):
    
    def __init__(self, path_to_weights):
        super().__init__(resnet.Bottleneck, [3, 4, 23, 3], path_to_weights)
    
    
class ResNet152Encoder(ResNetEncoder):
    
    def __init__(self, path_to_weights):
        super().__init__(resnet.Bottleneck, [3, 8, 36, 3], path_to_weights)

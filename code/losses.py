import torch


class BCERobustInstanceLoss(torch.nn.Module):
    """Robust binary cross entropy loss.
    
    Clip probabilities from edge values by 1e-7.
    
    Returns loss for each instance as a one dimensional pytorch tensor.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        loss = y_true * pixel_weights * torch.log(y_pred) \
                + (1 - y_true) * pixel_weights * torch.log(1. - y_pred)
        return -loss.mean(dim=1).mean(dim=1)
    
    
class BCERobustLoss(BCERobustInstanceLoss):
    """Robust binary cross entropy loss.
    
    Clip probabilities from edge values by 1e-7.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        return super().forward(y_pred, y_true, pixel_weights).mean()
    
    
class FocalRobustInstanceLoss(torch.nn.Module):
    """Focal loss for dense object detection.
    
    Clip probabilities from edge values by 1e-7.
    
    Returns loss for each instance as a one dimensional pytorch tensor.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    
    See https://arxiv.org/pdf/1708.02002.pdf for more details.
    """
    
    def __init__(self, gamma=2.0, alpha=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        y_true = y_true.view(-1, 1).long()
        
        y_pred = y_pred.view(-1, 1)
        y_pred = torch.cat((1 - y_pred, y_pred), 1)
        
        select = torch.FloatTensor(len(y_pred), 2).zero_().cuda()
        select.scatter_(1, y_true, 1.)
        
        y_pred = (y_pred * select).sum(1).view(-1, 1)
        y_pred = torch.clamp(y_pred, 1e-8, 1 - 1e-8)
        
        focus = torch.pow((1 - y_pred), self.gamma)
        focus = torch.where(focus < 1 - self.alpha, 
                            focus, 
                            torch.zeros(focus.size()).cuda())
        
        return -focus * y_pred.log()


class FocalRobustLoss(FocalRobustInstanceLoss):
    """Focal loss for dense object detection.
    
    Clip probabilities from edge values by 1e-7.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    
    See https://arxiv.org/pdf/1708.02002.pdf for more details.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        return super().forward(y_pred, y_true, pixel_weights).mean()

    
class SoftJaccardInstanceCoef(torch.nn.Module):
    """Differentiable extension of Jaccard coefficient (intersection over union).
    
    Returns loss for each instance as a one dimensional pytorch tensor.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    
    See https://arxiv.org/pdf/1801.05746.pdf (refered as J) for more details.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        I = (y_pred * y_true * pixel_weights).sum(dim=(1, 2))
        U = (y_pred * pixel_weights).sum(dim=(1, 2)) \
                + (y_true * pixel_weights).sum(dim=(1, 2)) - I
        IoU = (1e-5 + I) / (1e-5 + U)
        return IoU
    
    
class SoftJaccardCoef(SoftJaccardInstanceCoef):
    """Differentiable extension of Jaccard coefficient (intersection over union).
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    
    See https://arxiv.org/pdf/1801.05746.pdf (refered as J) for more details.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weihgts=1):
        return super().forward(y_pred, y_true, pixel_weights).mean()

    
class SoftJaccardInstanceBiclassCoef(torch.nn.Module):
    """Differentiable extension of Jaccard coefficient (intersection over union) for
    two class: object and background.
    
    Returns loss for each instance as a one dimensional pytorch tensor.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    
    See https://arxiv.org/pdf/1801.05746.pdf (refered as J) for more details.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        I1 = (y_pred * y_true * pixel_weights).sum(dim=(1, 2))
        U1 = (y_pred * pixel_weights).sum(dim=(1, 2)) \
                + (y_true * pixel_weights).sum(dim=(1, 2)) - I1
        IoU1 = (1e-5 + I1) / (1e-5 + U1)
        
        I2 = ((1. - y_pred) * (1 - y_true) * pixel_weights).sum(dim=(1, 2))
        U2 = ((1. - y_pred) * pixel_weights).sum(dim=(1, 2)) \
                + ((1 - y_true) * pixel_weights).sum(dim=(1, 2)) - I2
        IoU2 = (1e-5 + I2) / (1e-5 + U2)
        
        return (IoU1 + IoU2) / 2
    
    
class SoftJaccardBiclassCoef(SoftJaccardInstanceBiclassCoef):
    """Differentiable extension of Jaccard coefficient (intersection over union) for
    two class: object and background.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    
    See https://arxiv.org/pdf/1801.05746.pdf (refered as J) for more details.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weihgts=1):
        return super().forward(y_pred, y_true, pixel_weihgts).mean()
    
    
class SoftDiceInstanceCoef(torch.nn.Module):
    """Differentiable extension of Dice coefficient.
    
    Returns loss for each instance as a one dimensional pytorch tensor.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        I = (y_pred * y_true * pixel_weights).sum(dim=(1, 2))
        U = (y_pred * pixel_weights).sum(dim=(1, 2)) \
                + (y_true * pixel_weights).sum(dim=(1, 2))
        IoU = (1e-5 + I) / (1e-5 + U)
        return IoU
    
    
class SoftDiceCoef(SoftDiceInstanceCoef):
    """Differentiable extension of Dice coefficient.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        return super().forward(y_pred, y_true, pixel_weights).mean()

    
class SoftDiceInstanceBiclassCoef(torch.nn.Module):
    """Differentiable extension of Dice coefficient for two class: object and background.
    
    Returns loss for each instance as a one dimensional pytorch tensor.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        I1 = (y_pred * y_true * pixel_weights).sum(dim=(1, 2))
        U1 = (y_pred * pixel_weights).sum(dim=(1, 2)) \
                + (y_true * pixel_weights).sum(dim=(1, 2))
        IoU1 = (1e-5 + I1) / (1e-5 + U1)
        
        I2 = ((1. - y_pred) * (1 - y_true) * pixel_weights).sum(dim=(1, 2))
        U2 = ((1. - y_pred) * pixel_weights).sum(dim=(1, 2)) \
                + ((1 - y_true) * pixel_weights).sum(dim=(1, 2))
        IoU2 = (1e-5 + I2) / (1e-5 + U2)
        
        return (IoU1 + IoU2) / 2
    

class SoftDiceBiclassCoef(SoftDiceInstanceBiclassCoef):
    """Differentiable extension of Dice coefficient for two class: object and background.
    
    Parameters:
        y_pred: pytorch tensor of shape (batch_size, n_channles, height, width)
            probabilities for each pixel
        y_true: pytorch tensor of shape (batch_size, n_channles, height, width)
            true lablels for each pixel
        pixel_weights: pytorch tensor of shape (batch_size, n_channles, height, width)
            weihgts for per pixel losses
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, pixel_weights=1):
        return super().forward(y_pred, y_true, pixel_weights).mean()

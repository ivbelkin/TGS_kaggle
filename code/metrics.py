import numpy as np


def instanceSoftIoU2d(pred_masks, true_masks):
    """Calculate intersection over union for every instance
    
    Parameters:
        masks: numpy array of shape (batch_size, height, width)
            predicted binary masks
        true_masks: numpy array of shape (batch_size, height, width)
            groung truth masks
    
    Returns:
        numpy array of shape (batch_size,)
    """
    I = (pred_masks * true_masks).sum(axis=(1, 2))
    U = pred_masks.sum(axis=(1, 2)) + true_masks.sum(axis=(1, 2)) - I
    idx = U == 0
    I[idx] = 1
    U[idx] = 1
    return I / U


def meanSoftIoU2d(pred_masks, true_masks):
    """Calculate mean intersection over union
    
    Parameters:
        masks: numpy array of shape (batch_size, height, width)
            predicted binary masks
        true_masks: numpy array of shape (batch_size, height, width)
            groung truth masks
    
    Returns:
        float
    """
    iou = instanceSoftIoU2d(pred_masks, true_masks)
    return np.mean(iou)


def instanceIoU2d(pred_masks, true_masks, treashold=0.5):
    pred_masks_bin = binarize(pred_masks, treashold)
    return instanceSoftIoU2d(pred_masks_bin, true_masks)


def meanIoU2d(pred_masks, true_masks, treashold=0.5):
    pred_masks_bin = binarize(pred_masks, treashold)
    return meanSoftIoU2d(pred_masks_bin, true_masks)


def instanceAP2d(y_pred, y_true, bintreashold=0.5, treasholds=np.arange(0.5, 1.0, 0.05)):
    """Average precision metric at different treasholds for each instance
    
    Note:
        It was used as a leaderboar metric in TGS Salt Itentification Challenge
        
    See: https://www.kaggle.com/c/tgs-salt-identification-challenge#evaluation
    for details.
    """
    pred_masks = binarize(y_pred, bintreashold)
    iou = instanceIoU2d(pred_masks, y_true)
    S = np.zeros_like(iou, dtype=np.int32)
    for t in treasholds:
        S += (iou >= t).astype(np.int32)
    return S / len(treasholds)


def meanAP2d(y_pred, y_true, bintreashold=0.5, treasholds=np.arange(0.5, 1.0, 0.05)):
    """Mean average precision metric
    
    Note:
        It was used as a leaderboar metric in TGS Salt Itentification Challenge
        
    See: https://www.kaggle.com/c/tgs-salt-identification-challenge#evaluation
    for details.
    """
    ap = instanceAP2d(y_pred, y_true, bintreashold, treasholds)
    return np.mean(ap)


def instanceAccuracy2d(y_pred, y_true, treashold=0.5):
    pred_masks = binarize(y_pred, treashold)
    return (pred_masks == y_true).mean(axis=(1, 2))


def meanAccuracy2d(y_pred, y_true, treashold=0.5):
    acc = instanceAccuracy2d(y_pred, y_true, treashold)
    return np.mean(acc)


def binarize(proba, t):
    """Binarize belief maps at treashold t
    
    Parameters:
        proba: numpy array
            belief map
        t: float
            binarization treashold
    """
    return (proba > t).astype(np.float32)

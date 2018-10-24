import numpy as np
import os

from skimage import io
from sklearn.neighbors import KDTree


def read_image(path):
    """Read an one channel image as float (0-1) numpy array
    """
    image = io.imread(path, dtype=np.uint8)[:,:,0]
    image = image.astype(np.float32) / 255.
    return image


def read_mask(path):
    """Read an one channel maks as float binary numpy array
    """
    mask = io.imread(path, dtype=np.uint8)
    mask = np.where(mask > 128, 1, 0)
    mask = mask.astype(np.float32)
    return mask


def stack_ones_mask(mask):
    """Add ones as second channel of mask
    """
    ones = np.ones_like(mask)
    mask = np.stack((mask, ones), axis=2)
    return mask


def get_weights(mask, weight_fn):
    """Compute weights for each pixel based on its distance from separating line
    
    Parameters:
        mask: binary numpy array of shape (H, W)
        weight_fn: function, callable object
            function receives distance from point to separating line
            and returns weight of this point
            
    Returns:
        weights: float32 numpy array of shape (H, W)
    """
    weights = np.ones_like(mask, dtype=np.float32)
    if mask.std() == 0:
        return weights
    else:
        X, Y = np.meshgrid(np.arange(101), np.arange(101))
        points = np.stack((Y, X))

        tree = KDTree(points[:,mask > 0.5].T)
        pp = points[:,mask < 0.5]
        dists = tree.query(pp.T)[0][:,0]
        weights[pp[0], pp[1]] = weight_fn(dists)

        tree = KDTree(points[:,mask < 0.5].T)
        pp = points[:,mask > 0.5]
        dists = tree.query(pp.T)[0][:,0]
        weights[pp[0], pp[1]] = weight_fn(dists)

        return weights
    
    
def rle(image):
    """Perform run length encoding of a binary image
    
    Reutrns:
        image_rle: list of ints
    """
    image = list(image.T.flatten()) + [0]
    image_rle = []
    start = 0
    length = 0
    for i, px in enumerate(image):
        if px == 0:
            if length > 0:
                image_rle.extend([start + 1, length])
            start = i + 1
            length = 0
        else:
            length += 1
    return image_rle


def create_path_if_not_exists(path):
    """Create directory and all intermediate directories if not exist.
    
    Parameters:
        path: str
            absolute or relative path
    """
    folders = path.split("/")
    for i in range(len(folders)):
        path = os.path.join(*folders[:i + 1])
        if not os.path.exists(path):
            os.mkdir(path)

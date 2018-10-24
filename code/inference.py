import os
import torch
import numpy as np

from code.utils import *
from code.train import *


def predict_and_save(model, dataloader, keys, namekey, path, verbose=0):
    """Get predictions of model and save them to disk.
    
    Parameters:
        model: pytorch model
            '__call__' method of model must accept a dict returned
            as a batch by dataloader and returns dict
        dataloader: iterable
            returns batch as a dict of tensors
        keys: list
            keys of dict returned by model which will be saved
        namekey: str
            key of batch dict which will be used as a name for
            each instance
        path: str
            folder where to save predictions
        verbose: int
            if > 0 then output a progress bar
    """
    
    for key in keys:
        folder = os.path.join(path, key)
        create_path_if_not_exists(folder)
        
    dataloader = wrap_dataloader(dataloader, verbose)
    for inputs in dataloader:
        names = inputs[namekey]
        inputs = place_to(model.device, inputs)
        inputs_size = len(inputs[list(inputs.keys())[0]])
        outputs = model(**inputs)
        outputs = place_to(torch.device("cpu"), outputs)
        
        for key in keys:
            data = outputs[key].detach().numpy()
            folder = os.path.join(path, key)
            
            for instance, name in zip(data, names):
                filename = os.path.join(folder, name)
                np.save(filename, instance)
        
        free_tensors(inputs)
        free_tensors(outputs)


def load_pred_probs(path, verbose=0):
    """Load numpy arrays saved in folder and returns them with ids.
    """
    filenames = os.listdir(path)
    if verbose > 0:
        filenames = tqdm_notebook(filenames)
        
    ids, probs = [], []
    for fname in filenames:
        id_ = fname.split(".")[0]
        proba = np.load(os.path.join(path, fname))
        ids.append(id_)
        probs.append(proba)
        
    probs = np.stack(probs)
    return probs, ids


def force_zero_empty(path_to_images, ids, masks, verbose=0):
    """Set to zero mask of empty images.
    
    Parameters:
        path_to_images: str
        ids: list of str
            mask ids
        masks: numpy array of shape (len(ids), H, W)
        
    Returns:
        masks: numpy array of shape (len(ids), H, W)
    """
    if verbose > 0:
        ids = tqdm_notebook(ids)
    for i, id_ in enumerate(ids):
        imgname = "{}.png".format(id_)
        path = os.path.join(path_to_images, imgname)
        img = read_image(path)
        if (img == 0).all():
            masks[i] = np.zeros_like(masks[i])
    return masks


def prepare_submit(preds, ids, path, verbose=0):
    """Make and save submission file.
    
    Parameters:
        preds: integer numpy array of shape (len(ids), W, H)
        ids: list of str
        path: str
            filename to save submission file
    """
    if verbose > 0:
        preds = tqdm_notebook(preds)
    preds_rle = []
    for pred in preds:
        preds_rle.append(rle(pred))
    preds_rle = list(map(lambda x: " ".join(map(str, x)), preds_rle))
    submission = pd.DataFrame({
        "id": ids,
        "rle_mask": preds_rle,
    })
    submission.to_csv(path, index=False)

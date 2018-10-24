import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook
from skimage import io
from copy import deepcopy

from torch.utils.data import Dataset

from code.utils import *
from code.configs import *


class TGSDataset(Dataset):
    """Simple dataloader
    
    Parameters:
        path: string
            path to folder with images and masks(optinally)
        path_to_depths: string
            path to depths.csv
        progress_bar: bool
            if True, print progress bar while loading data
    
    Returns a dict with keys:
        id, image, mask, depth
    """
    
    def __init__(self, paths, path_to_depths, progress_bar=False):
        self.paths = paths
        self.path_to_depths = path_to_depths
        self.progress_bar = progress_bar
        
        self._load_data(self.paths)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        return self.data[idx]
    
    def _load_data(self, paths):
        self.data = []
        self.depths_df = pd.read_csv(self.path_to_depths, index_col="id")
        
        for path in paths:
            images_dir = os.path.join(path, "images")
            masks_dir = os.path.join(path, "masks")

            filenames = os.listdir(images_dir)
            load_masks = os.path.exists(masks_dir)

            if self.progress_bar:
                filenames = tqdm_notebook(filenames)

            for fname in filenames:
                id_ = fname.split(".")[0]

                image_name = os.path.join(images_dir, fname)
                image = read_image(image_name)

                depth = self.depths_df.loc[id_,"z"]

                record = {
                    "id": id_,
                    "image": image,
                    "depth": depth,
                }

                if load_masks:
                    mask_name = os.path.join(masks_dir, fname)
                    mask = read_mask(mask_name)
                    record["mask"] = mask

                self.data.append(record)

            
class TGSAugDataset(TGSDataset):
    """Simple dataloader with augmentations
    
    Parameters:
        augmenter: function
            receives (image, mask) and returns agmented versions of these two
    
    Returns a dict with keys:
        id, image, mask, depth
    """
    
    def __init__(self, augmenter=None, **kwargs):
        super().__init__(**kwargs)
        self.augmenter = augmenter
        
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        record = deepcopy(self.data[idx])
        record = self._apply_augmenter_if_can(record)
        return record

    def _apply_augmenter_if_can(self, record):
        if self.augmenter is not None:
            if "mask" in record:
                record["image"], record["mask"] = self.augmenter(record["image"], record["mask"])
            else:
                record["image"], _ = self.augmenter(record["image"], record["image"])
        return record

    
class TGSTTADataset(TGSDataset):
    """Dataset with test time augmentations
    
    Parameters:
        postproc: function
            receives (image, mask) and returns postprocessed versions of these two
    
    Returns a dict with keys:
        id, image, mask, depth
    """
    
    def __init__(self, postproc=None, **kwargs):
        super().__init__(**kwargs)
        self.postproc = postproc
        
    def __len__(self):
        return 2 * super().__len__()
     
    def __getitem__(self, idx):
        tp = idx // len(self.data)
        idx = idx % len(self.data)
        record = deepcopy(self.data[idx])
        record = self._apply(record, tp)
        return record

    def _apply(self, record, tp):
        if tp == 1:
            record["image"] = record["image"][:,::-1]
            if "mask" in record:
                record["mask"] = record["mask"][:,::-1]
            record["id"] += "_flipped"
        if self.postproc is not None:
            if "mask" in record:
                record["image"], record["mask"] = self.postproc(record["image"], record["mask"])
            else:
                record["image"], _ = self.postproc(record["image"], record["image"])
        return record


class TGSPairsDataset(TGSDataset):
    """Simple dataloader with augmentations
    
    Parameters:
        augmenter: function
            receives (image, mask) and returns agmented versions of these two
    
    Returns a dict with keys:
        id, image, mask, depth
    """
    
    def __init__(self, pairs_path, augmenter=None, **kwargs):
        super().__init__(**kwargs)
        self.augmenter = augmenter
        
        ids = []
        for r in self.data:
            ids.append(r["id"])
        
        pairs = list(pd.read_csv(pairs_path, header=None).to_records(index=False))
        for l,r in pairs:
            try:
                i = ids.index(l)
                j = ids.index(r)

                image = np.hstack((self.data[i]["image"], self.data[j]["image"]))
                if "mask" in self.data[i]:
                    mask = np.hstack((self.data[i]["mask"], self.data[j]["mask"]))

                for disp in [50]:
                    record = {
                        "id": self.data[i]["id"] + "_" + self.data[j]["id"],
                        "image": image[:,disp:disp+101],
                        "depth": (self.data[i]["depth"]+ self.data[j]["depth"]) / 2,
                    }
                    if "mask" in self.data[i]:
                        record["mask"] = mask[:,disp:disp+101]
                    self.data.append(record)
            except ValueError:
                pass
        
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        record = deepcopy(self.data[idx])
        record = self._apply_augmenter_if_can(record)
        return record

    def _apply_augmenter_if_can(self, record):
        if self.augmenter is not None:
            if "mask" in record:
                record["image"], record["mask"] = self.augmenter(record["image"], record["mask"])
            else:
                record["image"], _ = self.augmenter(record["image"], record["image"])
        return record
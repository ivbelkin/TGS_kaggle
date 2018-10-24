import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from skimage import io
from code.configs import *

np.warnings.filterwarnings("ignore")


def create_folders(cv_path, n_fold):
    """Create folders for different folds
    """
    os.mkdir(cv_path)
    for i in range(n_fold):
        dname = "fold-{}".format(i + 1)

        os.mkdir(os.path.join(cv_path, dname))

        os.mkdir(os.path.join(cv_path, dname, "train"))
        os.mkdir(os.path.join(cv_path, dname, "train", "images"))
        os.mkdir(os.path.join(cv_path, dname, "train", "masks"))

        os.mkdir(os.path.join(cv_path, dname, "valid"))
        os.mkdir(os.path.join(cv_path, dname, "valid", "images"))
        os.mkdir(os.path.join(cv_path, dname, "valid", "masks"))
        

def filter_images(depths_df, path):
    """Remains only images presented in '`path`/images'
    """
    train_ids = [fname.split(".")[0] 
                 for fname in os.listdir(os.path.join(path, "images"))]
    depths_df["presented"] = np.in1d(depths_df["id"], train_ids)
    depths_df = depths_df.query("presented == True").drop("presented", axis=1)
    return depths_df


def add_salt_coverage_column(depths_df, path):
    """Add salt coverage used masks in '`path`/masks'
    """
    depths_df["salt"] = 0
    for i, id_ in enumerate(tqdm(depths_df["id"])):
        p = os.path.join(path, "masks", id_ + ".png")
        mask = io.imread(p, dtype=np.uint8)
        mask = (mask > 128).astype(np.uint8)
        depths_df["salt"].iloc[i] = mask.sum()
    return depths_df


def filter_empty_images(depths_df, path):
    """Remains only not empty images in '`path`/images'
    """
    depths_df["empty"] = False
    for i, id_ in enumerate(tqdm(depths_df["id"])):
        p = os.path.join(path, "images", id_ + ".png")
        img = io.imread(p, dtype=np.uint8)
        depths_df["empty"].iloc[i] = (img == 0).all()
    depths_df = depths_df.query("empty == False").drop("empty", axis=1)
    return depths_df


def make_random_folds(depths_df, n_fold, random_state):
    """Shuffle images and add fold column
    """
    ids = np.arange(len(depths_df))
    np.random.seed(random_state)
    np.random.shuffle(ids)
    depths_df = depths_df.iloc[ids]
    depths_df["fold"] = 1 + np.arange(len(depths_df)) % n_fold
    return depths_df


def make_depth_stratified_folds(depths_df, n_fold):
    """Sort by depth and add fold column
    """
    depths_df = depths_df.sort_values("z")
    depths_df["fold"] = 1 + np.arange(len(depths_df)) % n_fold
    return depths_df


def make_salt_stratified_folds(depths_df, n_fold):
    """Sort by salt coverage and add fold column
    """
    depths_df = depths_df.sort_values("salt")
    depths_df["fold"] = 1 + np.arange(len(depths_df)) % n_fold
    return depths_df


def make_symlinks(depths_df, path_to_train, cv_path, n_fold):
    """Make symlinks for images and masks for folds in cross validation
    """
    for i in range(n_fold):
        dname = "fold-{}".format(i + 1)
        train_ids = depths_df.query("fold != {}".format(i + 1))["id"]
        valid_ids = depths_df.query("fold == {}".format(i + 1))["id"]

        for id_ in train_ids:
            src = os.path.join(os.path.join(path_to_train, "images"), id_ + ".png")
            dst = os.path.join(cv_path, dname, "train", "images", id_ + ".png")
            os.link(src, dst)

            src = os.path.join(os.path.join(path_to_train, "masks"), id_ + ".png")
            dst = os.path.join(cv_path, dname, "train", "masks", id_ + ".png")
            os.link(src, dst)

        for id_ in valid_ids:
            src = os.path.join(os.path.join(path_to_train, "images"), id_ + ".png")
            dst = os.path.join(cv_path, dname, "valid", "images", id_ + ".png")
            os.link(src, dst)

            src = os.path.join(os.path.join(path_to_train, "masks"), id_ + ".png")
            dst = os.path.join(cv_path, dname, "valid", "masks", id_ + ".png")
            os.link(src, dst)


if __name__ == "__main__":
    depths_df = pd.read_csv(PATH_TO_DEPTHS)
    print("Selecting train images...")
    depths_df = filter_images(depths_df, PATH_TO_TRAIN)
    print("Filtering empty images...")
    depths_df = filter_empty_images(depths_df, PATH_TO_TRAIN)
    print("Computing salt coverage...")
    depths_df = add_salt_coverage_column(depths_df, PATH_TO_TRAIN)
    
    print("Random folds...")
    path_to_cv = PATH_TO_RANDOM_CV
    n_fold = 5
    create_folders(path_to_cv, n_fold)
    depths_df = make_random_folds(depths_df, n_fold, 42)
    make_symlinks(depths_df, PATH_TO_TRAIN, path_to_cv, n_fold)
    
    print("Salt stratified folds...")
    path_to_cv = PATH_TO_SALT_CV
    n_fold = 5
    create_folders(path_to_cv, n_fold)
    depths_df = make_salt_stratified_folds(depths_df, n_fold)
    make_symlinks(depths_df, PATH_TO_TRAIN, path_to_cv, n_fold)
    
    print("Depth stratified folds...")
    path_to_cv = PATH_TO_DEPTH_CV
    n_fold = 5
    create_folders(path_to_cv, n_fold)
    depths_df = make_depth_stratified_folds(depths_df, n_fold)
    make_symlinks(depths_df, PATH_TO_TRAIN, path_to_cv, n_fold)

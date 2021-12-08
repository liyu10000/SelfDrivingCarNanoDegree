import argparse
import glob
import os
import random
import shutil
import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # create folders
    sub_folders = ['train', 'val', 'test']
    for sub_folder in sub_folders:
        os.makedirs(os.path.join(destination, sub_folder), exist_ok=True)
    # split data (since test data is already provided in workspace, will only split train/val here)
    data_dir = os.path.join(source, 'training_and_validation')
    files = os.listdir(data_dir)
    random.shuffle(files)
    n = len(files)
    train_val_split_ratio = 0.2
    n_train = int(n * (1 - train_val_split_ratio))
    train_files = files[:n_train]
    val_files = files[n_train:]
    for f in train_files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(destination, 'train'))
    for f in val_files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(destination, 'val'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
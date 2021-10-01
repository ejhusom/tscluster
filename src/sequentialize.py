#!/usr/bin/env python3
"""Split data into sequences.

Prepare the data for input to a neural network. A sequence with a given history
size is extracted from the input data, and matched with the appropriate target
value(s).

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

from config import DATA_PATH, DATA_SEQUENTIALIZED_PATH, NON_DL_METHODS
from preprocess_utils import (
    find_files,
    flatten_sequentialized,
    split_cluster_sequences,
)


def sequentialize(dir_path):
    """Make sequences out of tabular data."""

    filepaths = find_files(dir_path, file_extension=".npy")
    print(filepaths)

    DATA_SEQUENTIALIZED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["sequentialize"]
    learning_method = yaml.safe_load(open("params.yaml"))["train"]["learning_method"]
    overlap = params["overlap"]
    pca = params["pca"]

    window_size = params["window_size"]


    for filepath in filepaths:

        X = np.load(filepath)

        # Split into sequences
        if window_size == 1:
            pass
        else:
            X = split_cluster_sequences(X, window_size, overlap=overlap)

        if params["shuffle_samples"]:
            permutation = np.random.permutation(X.shape[0])
            X = np.take(X, permutation, axis=0)

        # if pca:
            


        # Save X and y into a binary file
        np.savez(
            DATA_SEQUENTIALIZED_PATH
            / (os.path.basename(filepath).replace("scaled.npy", "sequentialized.npz")),
            X=X
        )


if __name__ == "__main__":

    sequentialize(sys.argv[1])

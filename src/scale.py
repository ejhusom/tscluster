#!/usr/bin/env python3
"""Scaling the inputs of the data set.

Possible scaling methods

TODO:
    Implement scaling when there is only one workout file.

Author:
    Erik Johannes Husom

Created:
    2020-09-16

"""
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import DATA_PATH, DATA_SCALED_PATH
from preprocess_utils import find_files


def scale(dir_path):
    """Scale training and test data.

    Args:
        dir_path (str): Path to directory containing files.

    """

    filepaths = find_files(dir_path, file_extension=".npy")

    DATA_SCALED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["scale"]
    input_method = params["input"]

    if input_method == "standard":
        scaler = StandardScaler()
    elif input_method == "minmax":
        scaler = MinMaxScaler()
    elif input_method == "robust":
        scaler = RobustScaler()
    elif input_method is None:
        scaler = StandardScaler()
    else:
        raise NotImplementedError(f"{input_method} not implemented.")

    data_overview = {}
    data_matrices = []

    for filepath in filepaths:

        X = np.load(filepath)
        data_overview[filepath] = {"X": X}
        data_matrices.append(X)

    X = np.concatenate(data_matrices)

    # Fit a scaler to the training data
    scaler = scaler.fit(X)

    for filepath in data_overview:

        # Scale inputs
        if input_method == None:
            X = data_overview[filepath]["X"]
        else:
            X = scaler.transform(data_overview[filepath]["X"])

        # Save X and y into a binary file
        np.save(
            DATA_SCALED_PATH
            / (
                os.path.basename(filepath).replace(
                    ".npy",
                    "-scaled.npy",
                )
            ),
            X,
        )


if __name__ == "__main__":

    np.random.seed(2020)

    scale(sys.argv[1])

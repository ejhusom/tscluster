#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combines workout files into one.

Author:   
    Erik Johannes Husom

Created:  
    2020-10-29

"""
import os
import sys

import numpy as np

from config import DATA_COMBINED_PATH
from preprocess_utils import find_files


def combine(dir_path):
    """Combine data from multiple input files into one dataset.

    Args:
        dir_path (str): Path to directory containing files.

    """

    filepaths = find_files(dir_path, file_extension=".npz")

    DATA_COMBINED_PATH.mkdir(parents=True, exist_ok=True)

    data = []

    for filepath in filepaths:
        infile = np.load(filepath)
        data.append(infile["X"])

    X = np.concatenate(data)

    np.savez(DATA_COMBINED_PATH / "X.npz", X=X)


if __name__ == "__main__":

    np.random.seed(2020)

    combine(sys.argv[1])

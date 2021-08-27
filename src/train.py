#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train deep learning model to estimate power from breathing data.


Author:
    Erik Johannes Husom

Created:
    2020-09-16  

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from joblib import dump
from sklearn import cluster
from tslearn.clustering import TimeSeriesKMeans

from config import (
    DATA_PATH,
    MODELS_FILE_PATH,
    MODELS_PATH,
    NON_DL_METHODS,
    PLOTS_PATH,
    TRAININGLOSS_PLOT_PATH,
)


def train(filepath):
    """Train model to estimate power.

    Args:
        filepath (str): Path to training set.

    """

    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["train"]
    learning_method = params["learning_method"]

    # Load training set
    train_data = np.load(filepath)

    X = train_data["X"]

    n_features = X.shape[-1]
    hist_size = X.shape[-2]

    # Build model
    if learning_method == "kmeans":
        # model = cluster.KMeans(n_clusters=2, random_state=0, n_init=50, max_iter=500)
        model = TimeSeriesKMeans(n_clusters=3, metric="dtw")
    else:
        raise NotImplementedError(f"Learning method {learning_method} not implemented.")

    model.fit(X)
    dump(model, MODELS_FILE_PATH)

    plt.figure()
    plt.plot(model.labels_)
    plt.show()
    # print(model.cluster_centers_)

    # for c in model.cluster_centers_:
    #     plt.figure()
    #     plt.plot(c)
    #     plt.show()



if __name__ == "__main__":

    np.random.seed(2020)

    train(sys.argv[1])

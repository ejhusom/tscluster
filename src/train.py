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
from tslearn.clustering import KernelKMeans, TimeSeriesKMeans

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
    n_clusters = params["n_clusters"]

    # Load training set
    data = np.load(filepath)

    X = data["X"]

    n_features = X.shape[-1]
    hist_size = X.shape[-2]

    # Build model
    if learning_method == "timeserieskmeans":
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
    elif learning_method == "kmeans":
        X = X.reshape(X.shape[0], X.shape[-1])
        model = cluster.KMeans(n_clusters=n_clusters, random_state=0, n_init=50, max_iter=500)
    elif learning_method == "hierarchical":
        X = X.reshape(X.shape[0], X.shape[-1])
        model = cluster.AgglomerativeClustering(n_clusters=n_clusters, memory="cache")
    elif learning_method == "kernelkmeans":
        model = KernelKMeans(n_clusters=n_clusters)
    else:
        raise NotImplementedError(f"Learning method {learning_method} not implemented.")

    model.fit(X)
    dump(model, MODELS_FILE_PATH)


if __name__ == "__main__":

    np.random.seed(2020)

    train(sys.argv[1])

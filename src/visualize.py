#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import load

from config import (
    DATA_PATH,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    NON_DL_METHODS,
    PLOTS_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_FILE_PATH
)

def visualize(model_filepath, data_filepath):
    """Visualize results of clustering.

    Args:
        model_filepath (str): Path to model.
        data_filepath (str): Path to data set.

    """

    model = load(model_filepath)
    data = np.load(data_filepath)
    X = data["X"]
    # print(X.shape)

    plt.figure()
    plt.plot(model.labels_)
    plt.savefig("test.png")
    plt.show()
    # print(model.cluster_centers_)

    plot_clusters_2d(model, X)

def plot_clusters_2d(model, X):

    PREDICTION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    centers = model.cluster_centers_

    for i, c in enumerate(model.cluster_centers_):

        # Find all sequences in cluster
        a = np.where(model.labels_ == i, True, False)
        b = X[a]

        n = len(c[:,0])
        cmap = cm.jet
        fig = plt.figure(figsize=(8,8))
        # ax = fig.add_subplot(projection="3d")
        colors = np.linspace(0, 10, n)

        plt.plot(c[:,0], alpha=0.5)
        # plt.plot(c[:,0], c=colors, cmap=cmap, alpha=0.5)

        for j in range(b.shape[0]):
            plt.plot(b[j,:,0], c="black", alpha=0.05)

        plt.savefig(PLOTS_PATH / f"tool_number_cluster_{i}.png")
        plt.show()


def plot_clusters_3d(model, X):

    PREDICTION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    centers = model.cluster_centers_

    for i, c in enumerate(model.cluster_centers_):

        # Find all sequences in cluster
        # x = np.w
        a = np.where(model.labels_ == i, True, False)
        b = X[a]


        # plt.figure()
        # plt.plot(c[:,0])

        # for i in range(b.shape[0]):
        #     plt.plot(b[i,:,0], "k-", alpha=0.05)
        # plt.show()

        n = len(c[:,0])
        cmap = cm.jet
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection="3d")
        colors = np.linspace(0, 10, n)

        # ax.scatter(c[:,0], c[:,1], c[:,2], c=colors, cmap=cmap, alpha=0.5)

        for j in range(b.shape[0]):
            ax.scatter(b[j,:,0], b[j,:,1], b[j,:,2], c="black", alpha=0.05)

        plt.savefig(PLOTS_PATH / f"spindle_movement_3d_cluster_{i}.png")
        plt.show()


if __name__ == "__main__":

    np.random.seed(2020)

    visualize(sys.argv[1], sys.argv[2])

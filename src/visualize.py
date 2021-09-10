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
    DATA_PATH_RAW,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    NON_DL_METHODS,
    PLOTS_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_FILE_PATH
)
from preprocess_utils import find_files

def visualize(model_filepath, data_filepath):
    """Visualize results of clustering.

    Args:
        model_filepath (str): Path to model.
        data_filepath (str): Path to data set.

    """

    model = load(model_filepath)
    data = np.load(data_filepath)
    X = data["X"]

    plt.figure()
    plt.plot(model.labels_)
    plt.savefig("test.png")
    plt.show()
    # print(model.cluster_centers_)

    # plot_clusters_2d(model, X)
    plot_data_and_labels(X, model.labels_)


def plot_data_and_labels(X, labels):

    dataset = yaml.safe_load(open("params.yaml"))["profile"]["dataset"]
    dir_path = str(DATA_PATH_RAW)

    if dataset is not None:
        dir_path += "/" + dataset

    filepaths = find_files(dir_path, file_extension=".csv")

    dfs = []

    for filepath in filepaths:
        df = pd.read_csv(filepath)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df = df.replace(16,0)
    df = df.replace(40,1)
    df = df.replace(41,2)
    df = df.replace(128,3)

    # pn = df["PN"].iloc[4400:]

    print(len(labels))
    print(len(pn))


    # n = X.shape[0] * X.shape[1]
    # print(X.shape)
    # print(len(labels))

    # l = []

    # for i in range(len(labels)):
    #     for j in range(X.shape[1]):
    #         l.append(labels[i])

    # print(len(l))
    # X = X.flatten()

    plt.figure()
    # plt.xlim(0, n)
    
    # plt.plot(X)
    plt.plot(labels, label="labels")
    plt.plot(pn, label="PN")

    plt.show()



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

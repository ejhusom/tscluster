#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean up data.

TODO: Remove features with high correlation

Author:
    Erik Johannes Husom

Created:
    2021-06-30

"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from config import DATA_CLEANED_PATH, DATA_PATH, PROFILE_PATH
from preprocess_utils import find_files


def clean(dir_path):
    """Clean up inputs.

    Args:
        dir_path (str): Path to directory containing files.

    """

    # Load parameters
    dataset = yaml.safe_load(open("params.yaml"))["profile"]["dataset"]
    params = yaml.safe_load(open("params.yaml"))
    combine_files = params["clean"]["combine_files"]

    # If no name of data set is given, all files present in 'assets/data/raw'
    # will be used.
    if dataset is not None:
        dir_path += "/" + dataset

    filepaths = find_files(dir_path, file_extension=".csv")

    DATA_CLEANED_PATH.mkdir(parents=True, exist_ok=True)

    # Find removable variables from profiling report
    removable_variables = parse_profile_warnings()

    dfs = []

    for filepath in filepaths:

        # Read csv
        df = pd.read_csv(filepath)

        # If the first column is an index column, remove it.
        if df.iloc[:, 0].is_monotonic:
            df = df.iloc[:, 1:]

        for column in removable_variables:
            del df[column]

        df.dropna(inplace=True)

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    if combine_files:
        combined_df.to_csv(
            DATA_CLEANED_PATH / (os.path.basename("data-cleaned.csv"))
        )
    else:
        for filepath, df in zip(filepaths, dfs):
            df.to_csv(
                DATA_CLEANED_PATH
                / (os.path.basename(filepath).replace(".", "-cleaned."))
            )


def parse_profile_warnings():
    """Read profile warnings and find which columns to delete.

    Returns:
        removable_variables (list): Which columns to delete from data set.

    """
    params = yaml.safe_load(open("params.yaml"))["clean"]
    correlation_metric = params["correlation_metric"]

    profile_json = json.load(open(PROFILE_PATH / "profile.json"))
    messages = profile_json["messages"]
    variables = list(profile_json["variables"].keys())
    correlations = profile_json["correlations"]["pearson"]

    removable_variables = []

    percentage_zeros_threshold = params["percentage_zeros_threshold"]
    input_max_correlation_threshold = params["input_max_correlation_threshold"]

    for message in messages:
        message = message.split()
        warning = message[0]
        variable = message[-1]

        if warning == "[CONSTANT]":
            removable_variables.append(variable)
            print(f"Removed variable '{variable}' because it is constant.")
        if warning == "[ZEROS]":
            p_zeros = profile_json["variables"][variable]["p_zeros"]
            if p_zeros > percentage_zeros_threshold:
                removable_variables.append(variable)
                print(
                    f"Removed variable '{variable}' because % of zeros exceeds {percentage_zeros_threshold*100}%."
                )
        if warning == "[HIGH_CORRELATION]":
            try:
                correlation_scores = correlations[variables.index(variable)]
                for correlated_variable in correlation_scores:
                    if (
                        correlation_scores[correlated_variable]
                        > input_max_correlation_threshold
                        and variable != correlated_variable
                        and variable not in removable_variables
                    ):

                        removable_variables.append(correlated_variable)
                        print(
                            f"Removed variable '{correlated_variable}' because of high correlation ({correlation_scores[correlated_variable]:.2f}) with variable '{variable}'."
                        )
            except:
                # Pandas profiling might not be able to compute correlation
                # score for some variables, for example some categorical
                # variables.
                pass
                # print(f"{variable}: Could not find correlation score.")

    removable_variables = list(set(removable_variables))

    return removable_variables


if __name__ == "__main__":

    clean(sys.argv[1])

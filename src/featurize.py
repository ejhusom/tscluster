#!/usr/bin/env python3
"""Clean up inputs and add features to data set.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype
from scipy.signal import find_peaks
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from config import DATA_FEATURIZED_PATH, DATA_PATH, PROFILE_PATH
from preprocess_utils import find_files, move_column


def featurize(dir_path):
    """Clean up inputs and add features to data set.

    Args:
        dir_path (str): Path to directory containing files.

    """

    # Load parameters
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    features = params["featurize"]["features"]
    remove_features = params["featurize"]["remove_features"]
    add_rolling_features = params["featurize"]["add_rolling_features"]
    rolling_window_size = params["featurize"]["rolling_window_size"]

    filepaths = find_files(dir_path, file_extension=".csv")

    DATA_FEATURIZED_PATH.mkdir(parents=True, exist_ok=True)

    # ===============================================
    # TODO: Automatic encoding of categorical input variables
    # Read all data to fit one-hot encoder
    dfs = []

    for filepath in filepaths:
        df = pd.read_csv(filepath, index_col=0)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # ct = ColumnTransformer([('encoder', OneHotEncoder(), [38])],
    #         remainder='passthrough')

    # ct.fit(combined_df)

    categorical_variables = find_categorical_variables()

    # Remove variables that was removed in the cleaning process
    categorical_variables = [
        var
        for var in categorical_variables
        if var in combined_df.columns
    ]

    print(combined_df)
    print(f"Cat: {categorical_variables}")
    print(combined_df[categorical_variables])

    column_transformer = ColumnTransformer(
        [("encoder", OneHotEncoder(), categorical_variables)], remainder="passthrough"
    )

    # combined_df = column_transformer.fit_transform(combined_df)

    # print(combined_df)
    # print(combined_df.shape)
    # print(combined_df[categorical_variables])

    # categorical_encoder = OneHotEncoder()
    # categorical_encoder.fit(combined_df)
    # ===============================================

    for filepath in filepaths:

        # Read csv
        df = pd.read_csv(filepath, index_col=0)

        # If no features are specified, use all columns as features
        if not isinstance(features, list):
            features = df.columns

        # Check if wanted features from params.yaml exists in the data
        for feature in features:
            if feature not in df.columns:
                print(f"Feature {feature} not found!")

        for col in df.columns:
            # Remove feature from input. This is useful in the case that a raw
            # feature is used to engineer a feature, but the raw feature itself
            # should not be a part of the input.
            if col not in features:
                print(f"Deleting column '{col}'")
                del df[col]

            # Remove feature if it is non-numeric
            elif not is_numeric_dtype(df[col]):
                print(f"Deleting column '{col}'")
                del df[col]

        if add_rolling_features:
            df = compute_rolling_features(
                df, rolling_window_size, ignore_columns=[]
            )

        # if type(remove_features) is list:
        #     for col in remove_features:
        #         del df[col]

        # # Save data
        # df.to_csv(
        #     DATA_FEATURIZED_PATH
        #     / (os.path.basename(filepath).replace(".", "-featurized."))
        # )
        np.save(
            DATA_FEATURIZED_PATH
            / os.path.basename(filepath).replace("cleaned.csv", "featurized.npy"),
            df.to_numpy(),
        )

    # Save list of features used
    input_columns = [col for col in df.columns]
    pd.DataFrame(input_columns).to_csv(DATA_PATH / "input_columns.csv")


def compute_rolling_features(df, window_size, ignore_columns=None):
    """
    This function adds features to the input data, based on the arguments
    given in the features-list.

    Available features (TODO):

    - mean
    - std
    - autocorrelation
    - abs_energy
    - absolute_maximum
    - absolute_sum_of_change

    Args:
    df (pandas DataFrame): Data frame to add features to.
    features (list): A list containing keywords specifying which features to
        add.

    Returns:
        df (pandas DataFrame): Data frame with added features.

    """

    columns = [col for col in df.columns if col not in ignore_columns]

    for col in columns:
        df[f"{col}_sum"] = df[col].rolling(window_size).sum()
        df[f"{col}_gradient"] = np.gradient(df[col])
        df[f"{col}_mean"] = df[col].rolling(window_size).mean()
        maximum = df[col].rolling(window_size).max()
        minimum = df[col].rolling(window_size).min()
        df[f"{col}_maximum"] = maximum
        df[f"{col}_minimum"] = minimum
        df[f"{col}_min_max_range"] = maximum - minimum
        slope = calculate_slope(df[col])
        # df[f"{col}_slope"] = slope
        # df[f"{col}_slope_sin"] = np.sin(slope)
        # df[f"{col}_slope_cos"] = np.cos(slope)
        df[f"{col}_standard_deviation"] = df[col].rolling(window_size).std()
        df[f"{col}_variance"] = np.var(df[col])
        df[f"{col}_peak_frequency"] = calculate_peak_frequency(df[col])

    df = df.dropna()
    return df


def calculate_peak_frequency(series, rolling_mean_window=200):

    peaks_indices = find_peaks(series, distance=5)[0]
    peaks = np.zeros(len(series))
    peaks[peaks_indices] = 1

    freq = []
    f = 0
    counter = 0

    for i, p in enumerate(peaks):

        if p == 1:
            f = 10 / counter
            counter = 0
        else:
            counter += 1

        freq.append(f)

    freq = pd.Series(freq).rolling(rolling_mean_window).mean()

    return freq


def calculate_slope(series, shift=2, rolling_mean_window=1, absvalue=False):
    """Calculate slope.

    Args:
        series (array): Data for slope calculation.
        shift (int): How many steps backwards to go when calculating the slope.
            For example: If shift=2, the slope is calculated from the data
            point two time steps ago to the data point at the current time
            step.
        rolling_mean_window (int): Window for calculating rolling mean.

    Returns:
        slope (array): Array of slope angle.

    """

    v_dist = series - series.shift(shift)
    h_dist = 0.1 * shift

    slope = np.arctan(v_dist / h_dist)

    if absvalue:
        slope = np.abs(slope)

    slope = slope.rolling(rolling_mean_window).mean()

    return slope


def find_categorical_variables():
    """Find categorical variables based on profiling report.

    Returns:
        categorical_variables (list): List of categorical variables.

    """

    params = yaml.safe_load(open("params.yaml"))["clean"]

    profile_json = json.load(open(PROFILE_PATH / "profile.json"))
    variables = list(profile_json["variables"].keys())
    correlations = profile_json["correlations"]["pearson"]

    categorical_variables = []

    for var in variables:

        try:
            n_categories = profile_json["variables"][var]["n_category"]
            categorical_variables.append(var)
        except:
            pass

    # categorical_variables = list(set(categorical_variables))

    # if target in categorical_variables:
    #     print("Warning related to target variable. Check profile for details.")
    #     categorical_variables.remove(target)

    return categorical_variables


if __name__ == "__main__":

    np.random.seed(2020)

    featurize(sys.argv[1])

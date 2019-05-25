"""
This module contains functions that help evaluate trained models.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split(df, label_colname, test_size, seed):
    """
    Splits a dataset into training and testing dataframes.
    """
    X = df.drop(columns=[label_colname]).values
    y = df[label_colname].values
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=seed)

    df_train = pd.DataFrame(np.column_stack((X_train, y_train)),
                            columns=df.columns)
    df_test = pd.DataFrame(np.column_stack((X_test, y_test)),
                            columns=df.columns)
    return df_train, df_test


def time_split(df, colname, begin, end, n_splits, remove_date=True):
    """
    Splits a dataframe by a given date column. The date range given by begin
    and end is divided into n_splits equal size splits. These splits are then
    used to generate a series of n_splits - 1 training and test sets.

    The training sets are cumulative in that each subsequent training set
    contains all earlier training sets. The test sets are always the most
    recent split.
    """
    window = (end - begin) / n_splits
    thresholds = [begin + window * i for i in range(1, n_splits + 1)]
    splits = \
        [(df[df[colname] <= train_set_end],
          df[(df[colname] > train_set_end) & (df[colname] <= test_set_end)])
         for train_set_end, test_set_end
         in zip(thresholds, thresholds[1:])]

    if remove_date:
        splits = [(df_train.drop(columns=[colname]),
                   df_test.drop(columns=[colname]))
                   for df_train, df_test in splits]

    return splits

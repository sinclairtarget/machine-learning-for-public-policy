"""
This module contains functions that explore distributions of values and
relationships between columns.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extrema(narray):
    """
    Returns a (min, max) tuple for the narry.
    """
    return (np.min(narray), np.max(narray))


def unique_columns(df):
    """
    Returns a list of columns that have a unique value for every row.
    """
    colnames = []
    for colname in df.columns:
        if df[colname].nunique() == len(df):
            colnames.append(colname)

    return colnames


def binary_columns(df):
    """
    Returns a list of columns in the dataframe that only have two unique
    values.
    """
    return [name for name in df.columns if len(pd.unique(df[name])) == 2]


def plot_missing(df, *colnames):
    """
    Plots a bar chart comparing the number of present and missing values for
    the given columns.
    """
    fig = plt.figure(1)

    for i, colname in enumerate(colnames, start=1):
        plt.subplot(1, len(colnames), i)
        missing = df[colname][df[colname].isnull()]
        present = df[colname][df[colname].notnull()]
        plt.title(colname)
        plt.bar(['present', 'missing'], [len(present), len(missing)])

    plt.tight_layout()
    plt.show()


def plot_label(df, colname):
    """
    Plots a bar chart showing the distribution of values between true and
    false for a single column.
    """
    plot_binary_predicate(df, colname, 'true', 'false', lambda col: col == True)


def plot_binary_predicate(df, col, true_name, false_name, predicate):
    """
    Plots a bar chart showing the distribution of values between two classes,
    one where the predicate is true and one where the predicate is false.
    """
    true_count = len(df[col][predicate(df[col])])
    false_count = len(df[col][~predicate(df[col])])
    plt.bar([true_name, false_name], [true_count, false_count])
    plt.show()

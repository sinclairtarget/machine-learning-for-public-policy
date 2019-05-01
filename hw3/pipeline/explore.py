"""
This module contains functions that explore distributions of values and
relationships between columns.
"""
import numpy as np
import matplotlib.pyplot as plt

def extrema(narray):
    """
    Returns a (min, max) tuple for the narry.
    """
    return (np.min(narray), np.max(narray))


def plot_missing(df, *cols):
    """
    Plots a bar chart comparing the number of present and missing values for
    the given columns.
    """
    fig = plt.figure(1)

    for i, col in enumerate(cols, start=1):
        plt.subplot(1, len(cols), i)
        missing = df[col][df[col].isnull()]
        present = df[col][df[col].notnull()]
        plt.title(col)
        plt.bar(['present', 'missing'], [len(present), len(missing)])

    plt.tight_layout()
    plt.show()

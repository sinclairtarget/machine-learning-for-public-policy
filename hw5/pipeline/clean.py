"""
This module contains helper functions for cleaning data and generating
features.
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing

def impute(df, col, how='avg'):
    """
    Replaces all null values in the named column with an imputed value.
    """
    subst = np.nan

    if how == 'avg':
        subst = np.average(df.loc[df[col].notnull(), col])
    else:
        raise Exception(f"\"how\" argument of \"{how}\" is not supported.")

    df.loc[df[col].isnull(), col] = subst


def dummify(df, *colnames):
    """
    Converts columns containing discrete values into several binary columns.
    """
    for colname in colnames:
        uniqs = df[colname].unique()
        for uniq in uniqs:
            new_colname = colname + '_is_' + str(uniq).lower().replace(' ', '_')
            df[new_colname] = (df[colname] == uniq).astype(float)

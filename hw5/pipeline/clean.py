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
    domains = {}
    for colname in colnames:
        domains[colname] = df[colname].unique()

    return dummify_domain(df, domains, *colnames)


def dummify_domain(df, domains, *colnames):
    """
    Converts columns containing discrete values into several binary columns.

    This version uses pre-existing category domains.
    """
    for colname in colnames:
        domain = domains[colname]
        unknown = colname + '_is_unknown'
        df[unknown] = 0.0

        uniqs = df[colname].unique()
        for uniq in uniqs:
            if uniq in domain:
                pretty_name = str(uniq).lower().replace(' ', '_')
                new_colname = colname + '_is_' + pretty_name
                df[new_colname] = (df[colname] == uniq).astype(float)
            else:
                df[unknown] = \
                    (df[unknown].astype(bool) | (df[colname] == uniq)).astype(float)

    return domains

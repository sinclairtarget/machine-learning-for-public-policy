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

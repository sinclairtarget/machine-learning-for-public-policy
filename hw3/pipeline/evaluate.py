"""
This module contains functions that help evaluate trained models.
"""
import pandas as pd

def accuracy(df, target_col, predict_col=None):
    """
    Calculates accuracy given a label/target column and a predicted value
    column.
    """
    if predict_col == None:
        predict_col = target_col + '_predict'

    total = len(df)
    correct = len(df[df[target_col] == df[predict_col]])
    return correct / total

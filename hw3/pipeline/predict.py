"""
This module contains helper methods that wrap sklearn prediction models.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from .result import PredictionResult

def dummy(df, label_colname, seed):
    """
    Returns a dummy classifier using the 'stratified' technique.
    """
    y = df[label_colname].values
    X = df.drop(columns=[label_colname]).values

    model = DummyClassifier(random_state=seed)
    model.fit(X, y)
    return model


def logistic_regression(dfs, label_colname, penalty='l2'):
    """
    Returns logistic regression models fitted to the given training data.
    """
    models = []
    for df in dfs:
        y = df[label_colname].values
        X = df.drop(columns=[label_colname]).values

        model = LogisticRegression(solver='liblinear', penalty=penalty)
        model.fit(X, y)
        models.append(model)

    return models


def test(model, df, label_colname):
    """
    Uses the fitted model to generate a prediction result dataframe.
    """
    y_actual = df[label_colname].values
    X = df.drop(columns=[label_colname]).values
    y_predict = model.predict(X)

    df_results = pd.DataFrame({ 'actual': y_actual,
                                'predict': y_predict },
                              dtype=float)
    return PredictionResult(df_results)


def test_all(models, dfs, label_colname):
    """
    Generates prediction results a series of models match with test data.
    """
    return PredictionResult.stack([
        test(model, df, label_colname)
        for model, df in zip(models, dfs)
    ])

"""
This module contains helper methods that wrap sklearn prediction models.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .result import PredictionResult

def logistic_regression(df, label_colname):
    """
    Returns a logistic regression model fitted to the given data.
    """
    y = df[label_colname].values
    X = df.drop(columns=[label_colname]).values

    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    return model


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

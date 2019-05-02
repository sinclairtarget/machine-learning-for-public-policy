"""
This module contains helper methods that wrap sklearn prediction models.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from .result import PredictionResult

class Trainer:
    """
    Provides model training methods for a particular set of training data.
    """
    def __init__(self, dfs, label_colname, seed):
        self.dfs = dfs
        self.label_colname = label_colname
        self.seed = seed


    def dummy(self):
        """
        Returns dummy classifier models using the 'stratified' technique.
        """
        models = []
        for X, y in self._training_data():
            model = DummyClassifier(random_state=self.seed)
            model.fit(X, y)
            models.append(model)

        return models


    def logistic_regression(self, penalty='l2'):
        """
        Returns logistic regression models fitted to the training data.
        """
        models = []
        for X, y in self._training_data():
            model = LogisticRegression(solver='liblinear',
                                       penalty=penalty,
                                       random_state=self.seed)
            model.fit(X, y)
            models.append(model)

        return models


    def decision_tree(self, max_depth=None):
        """
        Returns decision tree models fitted to the given training data.
        """
        models = []
        for X, y in self._training_data():
            model = DecisionTreeClassifier(max_depth=max_depth,
                                           random_state=self.seed)
            model.fit(X, y)
            models.append(model)

        return models


    def _training_data(self):
        for df in self.dfs:
            X = df.drop(columns=[self.label_colname]).values
            y = df[self.label_colname].values
            yield X, y


class Tester:
    """
    Provides test methods for a particular set of test data.
    """
    def __init__(self, dfs, label_colname):
        self.dfs = dfs
        self.label_colname = label_colname


    def test(self, *models):
        """
        Uses the fitted models to generate a prediction result dataframe.
        """
        if len(models) != len(self.dfs):
            raise Exception('Number of models does not match test sets.')

        results = []
        for (X, y_actual), model in zip(self._test_data(), models):
            y_predict = model.predict(X)
            df_results = pd.DataFrame({ 'actual': y_actual,
                                        'predict': y_predict },
                                      dtype=float)
            results.append(PredictionResult(df_results))

        return PredictionResult.stack(results)


    def _test_data(self):
        for df in self.dfs:
            X = df.drop(columns=[self.label_colname]).values
            y_actual = df[self.label_colname].values
            yield X, y_actual

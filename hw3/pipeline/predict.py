"""
This module contains helper methods that wrap sklearn prediction models.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .result import PredictionResult, ResultCollection
from .util import wrap_list

class Trainer:
    """
    Provides model training methods for a particular set of training data.
    """
    def __init__(self, *dfs, label_colname=None, seed=None):
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

        return models if len(models) > 1 else models[0]


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

        return models if len(models) > 1 else models[0]


    def decision_tree(self, max_depth=None):
        """
        Returns decision tree models fitted to the training data.
        """
        models = []
        for X, y in self._training_data():
            model = DecisionTreeClassifier(max_depth=max_depth,
                                           random_state=self.seed)
            model.fit(X, y)
            models.append(model)

        return models if len(models) > 1 else models[0]


    def k_nearest(self, k=5):
        """
        Returns k-nearest neighbors models fitted to the training data.
        """
        models = []
        for X, y in self._training_data():
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X, y)
            models.append(model)

        return models if len(models) > 1 else models[0]


    def linear_svm(self, c=1):
        """
        Returns linear svm models fitted to the training data.
        """
        models = []
        for X, y in self._training_data():
            # Prefer dual=False when n_samples > n_features
            model = Pipeline([('scale', StandardScaler()),
                              ('svm', LinearSVC(C=c, dual=False,
                                                random_state=self.seed))])
            model.fit(X, y)
            models.append(model)

        return models if len(models) > 1 else models[0]


    def forest(self, n_trees=10):
        """
        Returns random forest models fitted to the training data.
        """
        models = []
        for X, y in self._training_data():
            model = RandomForestClassifier(n_estimators=n_trees,
                                           random_state=self.seed)
            model.fit(X, y)
            models.append(model)

        return models if len(models) > 1 else models[0]


    def bagging(self, n_estimators=10):
        """
        Returns bagging models fitted to the training data.

        Underlying base estimator is a decision tree.
        """
        models = []
        for X, y in self._training_data():
            model = BaggingClassifier(n_estimators=n_estimators,
                                      random_state=self.seed)
            model.fit(X, y)
            models.append(model)

        return models if len(models) > 1 else models[0]


    def train_all(self, parameters={}, exclude=[]):
        """
        Train all the things.

        Returns a dictionary with method names as keys and models as values.
        """
        methods = {
            'dummy': self.dummy,
            'logistic_regression': self.logistic_regression,
            'decision_tree': self.decision_tree,
            'k_nearest': self.k_nearest,
            'linear_svm': self.linear_svm,
            'forest': self.forest,
            'bagging': self.bagging
        }

        models = dict()

        for name, func in methods.items():
            if name in exclude:
                next

            params = parameters.get(name) or {}
            models[name] = func(**params)

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
    def __init__(self, *dfs, label_colname=None):
        self.dfs = dfs
        self.label_colname = label_colname


    def test(self, *models):
        """
        Uses the fitted models to generate a prediction result dataframe.

        Model results are "stacked": They are assumed to be the results of the
        same conceptual model applied to different splits.
        """
        results = self._test(*models)
        if len(results) > 1:
            return ResultCollection.from_stack(results)
        else:
            return results[0]


    def _test(self, *models):
        if len(models) != len(self.dfs):
            raise Exception('Number of models does not match test sets.')

        results = []
        for (X, y_actual), model in zip(self._test_data(), models):
            if isinstance(model, Pipeline) and \
               isinstance(model.named_steps.svm, LinearSVC):
                y_score = model.decision_function(X)
            else:
                y_score = model.predict_proba(X)[:,1]

            y_predict = model.predict(X)
            df_results = pd.DataFrame({ 'actual': y_actual,
                                        'score': y_score,
                                        'predict': y_predict },
                                      dtype=float)
            results.append(PredictionResult(df_results))

        return results


    def evaluate(self, model_dict, thresholds=None):
        """
        Tests lots of different models at different thresholds.
        """
        collection = ResultCollection()
        for name, models in model_dict.items():
            models = wrap_list(models)
            result = self._test(*models)[0] # Hack for now

            if thresholds:
                results = result.with_thresholds(thresholds)

                this_collection = ResultCollection.from_stack(results,
                                                              index=thresholds)
                collection.join(name, this_collection)
            else:
                collection.join(name, result)

        return collection


    def _test_data(self):
        for df in self.dfs:
            X = df.drop(columns=[self.label_colname]).values
            y_actual = df[self.label_colname].values
            yield X, y_actual

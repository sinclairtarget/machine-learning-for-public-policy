---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Predicting Funding for Projects on DonorsChoose.org
## Goal
Predict whether a project posted on DonorsChoose.org will _not_ get funded
within 60 days of first being posted on the site, perhaps so that we can
increase the visibility of the project by adding it to e.g. the site landing
page and thus get it funded.

Both precision and recall will be important to us. We want to try to make sure
all projects get funded, but we might also only have limited room on our site
landing page, so it is important that the projects we feature were really in
need of additional visibilty.

Since both precision and recall are important, we will focus on **F1 score** as
our evaluation metric.

## Environment Setup
```python
import pandas as pd
import pipeline
from pipeline import Trainer, Tester, ResultCollection
import cleaner

pipeline.notebook.set_up()

SEED = 1234

to_datetime = pipeline.datetime_converter('%m/%d/%y')
df = pipeline.read_csv('projects_2012_2013.csv',
                       converters={
                           'date_posted': to_datetime,
                           'datefullyfunded': to_datetime
                       })
df.head()
```

## Data Cleaning
### Remove Unnecessary Columns
First we remove any columns that contain a unique value for every row. These
columns will not be useful as features.

```python
pipeline.unique_columns(df)
```
We also drop all the other id columns, since we want to learn about the
_characteristics_ of successful projects rather than which teachers and schools
have successful projects.

Finally, we also drop all columns that contain geographic information other
than the longitude and latitude columns.

```python
%psource cleaner.unnecessary_columns
```
```python
df = cleaner.unnecessary_columns(df)
```

### Fix Types
Some columns do not have the correct types. For example, the `school_charter`
column contains `f` and `t` values but should instead be binary:
```python
pd.unique(df.school_charter)
```

These are all the implicitly binary columns:
```python
pipeline.binary_columns(df)
```

We convert all of them to explicit binary columns.
```python
%psource cleaner.fix_types
```
```python
df = cleaner.fix_types(df)
```

### Handle Missing Data
The following columns have missing data:
```python
missing = pipeline.missing_columns(df)
missing
```

```python
pipeline.plot_missing(df, *missing)
```

Given the large amount of missing data for both the `secondary_focus_subject`
and `secondary_focus_area` columns, we drop them. We then drop all remaining
rows with missing data, since there are so few.
```python
%psource cleaner.handle_missing
```
```python
df = cleaner.handle_missing(df)
```

### Handle Categorical Variables
We need to convert our categorical variables into a collection of binary dummy
variables.

These are our categorical variables:
```python
pipeline.categorical_columns(df)
```

We convert them all to binary dummy variables:
```python
%psource cleaner.handle_categorical
```
```python
df = cleaner.handle_categorical(df)
```

### Label Data
Finally, we need to label our data. We also need to sort it in order of
`date_posted` and drop our date columns.

```python
%psource cleaner.label
```
```python
label_colname = 'not_funded_in_60_days'
df = cleaner.label(df, label_colname)
```

Our final dataframe looks like the following:
```python
df.head()
```

We see below that 29% of projects are not funded within 60 days of being
posted.
```python
df.not_funded_in_60_days.value_counts(normalize=True)
```

## Parameter Selection
### Training Data
We separate our data into a validation and a test set. The validation set will
be used for experimenting with model parameters. The test set will be used to
evaluate our parameterized models at different thresholds.

```python
splits = pipeline.time_split(df, 3)
[(df_train.index.min(), df_train.index.max(), df_test.index.max()) \
    for df_train, df_test in splits]
```

Here we take only the first two splits to use for validation:
```python
validation_splits = splits[:-1]
holdout_split = splits[-1]
```

### Models
We will use our validation set to parameterize our models. This will give us
two validation training sets.
```python
dfs_train = [df_train for df_train, _ in validation_splits]
dfs_test = [df_test for _, df_test in validation_splits]

# Print training data sets
[('split' + str(i), df_train.index.min(), df_train.index.max()) \
    for i, df_train in enumerate(dfs_train, 1)]
```
```python
# Print test data sets
[('split' + str(i), df_test.index.min(), df_test.index.max()) \
    for i, df_test in enumerate(dfs_test, 1)]
```
```python
trainer = Trainer(*dfs_train, label_colname=label_colname, seed=SEED)
tester = Tester(*dfs_test, label_colname=label_colname)
```

#### Logistic Regression
For our logistic regression model, we have to decide whether to use L1 or L2
regularization.

```python
lr_results = ResultCollection()

for c in [1, 10, 100]:
    models = trainer.logistic_regression(c=c)
    lr_results.join('c_' + str(c), tester.test(*models))

lr_results.df
```
```python
lr_results.plot_statistic('f1')
```

It looks like using L1 regression here is marginally better, though both models
perform poorly.

#### Decision Tree
For our decision tree model, we need to experiment with tree depth to see what
depth is best.

```python
tree_results = ResultCollection()

tree_no_max_models = trainer.decision_tree(max_depth=None)
tree_results.join('no_max', tester.test(*tree_no_max_models))

for max_depth in [6, 12, 24]:
    models = trainer.decision_tree(max_depth=max_depth)
    tree_results.join(str(max_depth) + '_max', tester.test(*models))

tree_results.df
```
```python
tree_results.plot_statistic('f1')
```

It looks like setting no max depth is appropriate here. The tree's depth is
still limited by a min limit on the number of values per leaf node.

#### K-Nearest Neighbor
For our k-nearest neighbor model, we will have to experiment with setting
different values of $k$.

```python
k_nearest_results = ResultCollection()

for k in [3, 6, 12, 24]:
    models = trainer.k_nearest(k=k)
    k_nearest_results.join('k_' + str(k), tester.test(*models))

k_nearest_results.df
```
```python
k_nearest_results.plot_statistic('f1')
```

It looks like we want to keep $k$ on the smaller side.

#### SVM
For our support vector machine model, we will want to experiment with setting
$c$, which is a penalty term that should be lower when we have noisy data.

```python
svm_results = ResultCollection()

for c in [1, 10, 100]:
    models = trainer.linear_svm(c=c)
    svm_results.join('c_' + str(c), tester.test(*models))

svm_results.df
```
```python
svm_results.plot_statistic('f1')
```

It looks like choosing $c$ doesn't make much of a difference, so we might as
well go with the default of $c = 1$.

#### Random Forest
For random forests, we need to decide on the number of trees to use.

```python
forest_results = ResultCollection()

for n_trees in [100, 500, 1200]:
    models = trainer.forest(n_trees=n_trees)
    forest_results.join('n_' + str(n_trees), tester.test(*models))

forest_results.df
```
```python
forest_results.plot_statistic('f1')
```

Around 50 trees seems to be the best option.

#### Bagging
For the bagging approach, we need to decide on the number of base estimators to
create, or alternatively the number of bootstrap samples to draw.

```python
bagging_results = ResultCollection()

for n_estimators in [100, 500, 1200]:
    models = trainer.bagging(n_estimators=n_estimators)
    bagging_results.join('n_' + str(n_estimators), tester.test(*models))

bagging_results.df
```
```python
bagging_results.plot_statistic('f1')
```

About 100 estimators looks ideal.

#### Boosting
For the boosting approach, we again need to decide on the number of base
estimators.

```python
boosting_results = ResultCollection()

for n_estimators in [10, 50, 100]:
    models = trainer.boosting(n_estimators=n_estimators)
    boosting_results.join('n_' + str(n_estimators), tester.test(*models))

boosting_results.df
```
```python
boosting_results.plot_statistic('f1')
```

Again, about 100 estimators looks ideal.

## Evaluation
Now that we are evaluating our models, we will want to use our final holdout
time split.

```python
df_train, df_test = holdout_split

# Print final holdout training and test set
(df_train.index.min(), df_train.index.max(), df_test.index.max())
```
```python
trainer = Trainer(df_train, label_colname=label_colname, seed=SEED)
tester = Tester(df_test, label_colname=label_colname)
```

Next we set all of our model parameters according to the analysis we did in the
previous section and then train all of our models on our training set. We then
test our trained models on our holdout test set.

```python
params = {
    'linear_regression': { 'penalty': 'l1' },
    'decision_tree': { 'max_depth': None },
    'k_nearest': { 'k': 3 },
    'svm': { 'c': 1 },
    'forest': { 'n_trees': 50 },
    'bagging': { 'n_estimators': 100 },
    'boosting': { 'n_estimators': 100 }
}

models = trainer.train_all(parameters=params)
results = tester.evaluate(models)
results.df
```
```python
results.plot_statistic('f1', xlabel='model')
```

The above graph shows the F1 performance of our models with no threshold set.
This graph suggests that using either a decision tree or bagging is the best
approach.

We also want to test using our thresholds:
```python
threshold_results = \
    tester.evaluate(models, thresholds=[1, 2, 5, 10, 20, 30, 40, 50])
threshold_results.df
```
```python
threshold_results.plot_statistic('f1',
                                 xlabel='threshold')
```

This graph suggests that the boosting and bagging models are pretty good
overall. It is not clear to me why the logistic regression and linear svm
models, which were worse than the dummy model before, are now better.

We might also want to look at how our F1 metric breaks down into precision and
recall over our thresholds:
```python
threshold_results.plot_statistic('precision',
                                 xlabel='threshold')
```
```python
threshold_results.plot_statistic('recall',
                                 xlabel='threshold')
```

If we were targeting a specific threshold, then we would probably want to
maximize precision instead of F1. In that case, the above graph showing
precision would be important, and we might consider either the bagging or
boosting model as the clear winner, depending on the threshold.

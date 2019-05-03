---
jupyter:
  kernelspec:
    name: python3
    langauge: python
    display_name: Python 3
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

## Data Cleaning
```python
import pandas as pd
import pipeline

SEED = 1234

df = pipeline.read_csv('projects_2012_2013.csv')
```

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
df = df.drop(columns=['projectid',
                      'teacher_acctid',
                      'schoolid',
                      'school_ncesid',
                      'school_city',
                      'school_state',
                      'school_metro',
                      'school_district',
                      'school_county'])
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

We convert all of them to explicit binary columns:
```python
for colname in ['school_charter',
                'school_magnet',
                'eligible_double_your_impact_match']:
    df[colname] = (df[colname] == 't').astype(float)
```

The date columns should also be parsed into datetime objects.
```python
format = '%m/%d/%y'
df['date_posted'] = pd.to_datetime(df['date_posted'], format=format)
df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'], format=format)
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
and `secondary_focus_area` columns, we drop them:
```python
df = df.drop(columns=['secondary_focus_subject',
                      'secondary_focus_area'])
```

We then drop all remaining rows with missing data, since there are so few.
```python
df = df.dropna()
```

### Handle Categorical Variables
We need to convert our categorical variables into a collection of binary dummy
variables.

```python
pipeline.dummify(df, 'teacher_prefix',
                     'primary_focus_subject',
                     'primary_focus_area',
                     'resource_type',
                     'poverty_level',
                     'grade_level')
df = df.drop(columns=['teacher_prefix',
                      'primary_focus_subject',
                      'primary_focus_area',
                      'resource_type',
                      'poverty_level',
                      'grade_level'])
```

### Label Data
Finally, we need to label our data.

```python
label_colname = 'not_funded_in_60_days'
df[label_colname] = (
    (df.datefullyfunded - df.date_posted)
    .apply(lambda d: d.days > 60)
    .astype(float)
)
```

We also need to sort it in order of `date_posted`:
```python
df = df.sort_values('date_posted')
df = df.reset_index(drop=True)
```

We can now drop our date columns:
```python
df = df.drop(columns=['date_posted',
                      'datefullyfunded'])
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
from pipeline import Trainer, Tester

trainer = Trainer(*dfs_train, label_colname=label_colname, seed=SEED)
tester = Tester(*dfs_test, label_colname=label_colname)
```

#### Logistic Regression
For our logistic regression model, we have to decide whether to use L1 or L2
regularization.

```python
from pipeline import ResultCollection

lr_results = ResultCollection()
lr_l1_models = trainer.logistic_regression(penalty='l1')
lr_results.join('L1', tester.test(*lr_l1_models))

lr_l2_models = trainer.logistic_regression(penalty='l2')
lr_results.join('L2', tester.test(*lr_l2_models))
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

for c in [0.6, 0.8, 1]:
    models = trainer.linear_svm(c=c)
    svm_results.join('c_' + str(c), tester.test(*models))

svm_results.df
```
```python
svm_results.plot_statistic('f1')
```

It looks like choosing $c$ doesn't make much of a difference, so we might as
well go with the default of $c = 1$.

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
    'svm': { 'c': 1 }
}

models = trainer.train_all(parameters=params)
results = tester.evaluate(models)
results.df
```
```python
results.plot_statistic('f1', xlabel='model')
```

We also want to test using our thesholds:
```python
threshold_results = tester.evaluate(models,
                                    thresholds=[1, 2, 5, 10, 20, 30, 40, 50])
threshold_results.df
```
```python
threshold_results.plot_statistic('f1', xlabel='threshold')
```

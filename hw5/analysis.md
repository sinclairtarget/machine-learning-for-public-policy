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

We will focus on **AUC** as our evaluation metric when comparing models.

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
We will walk through the data cleaning process now using the entire dataset.
This is only to illustrate the transformations we do on the data—we will
repeat this cleaning step later for each of our training and test set pairs.

```python
df_example = df.copy()
```

### Remove Unnecessary Columns
First we remove any columns that contain a unique value for every row. These
columns will not be useful as features.

```python
pipeline.unique_columns(df_example)
```
We also drop all the other id columns, since we want to learn about the
_characteristics_ of successful projects rather than which teachers and schools
have successful projects.

Finally, we also drop some geographic columns that are too specific and would
result in too many features. Instead we rely on a combination of long/lat,
state, and metro to capture this information.

```python
%psource cleaner.unnecessary_columns
```
```python
df_example = cleaner.unnecessary_columns(df_example)
```

### Fix Types
Some columns do not have the correct types. For example, the `school_charter`
column contains `f` and `t` values but should instead be binary:
```python
pd.unique(df_example.school_charter)
```

These are all the implicitly binary columns:
```python
pipeline.binary_columns(df_example)
```

We convert all of them to explicit binary columns.
```python
%psource cleaner.fix_types
```
```python
df_example = cleaner.fix_types(df_example)
```

### Handle Missing Data
The following columns have missing data:
```python
missing = pipeline.missing_columns(df_example)
missing
```

```python
pipeline.plot_missing(df_example, *missing)
```

Given the large amount of missing data for both the `secondary_focus_subject`
and `secondary_focus_area` columns, we drop them. We then drop all remaining
rows with missing data, since there are so few.
```python
%psource cleaner.handle_missing
```
```python
df_example = cleaner.handle_missing(df_example)
```

### Handle Categorical Variables
We need to convert our categorical variables into a collection of binary dummy
variables.

These are our categorical variables:
```python
pipeline.categorical_columns(df_example)
```

We convert them all to binary dummy variables:
```python
%psource cleaner.handle_categorical
```
```python
df_example, _ = cleaner.handle_categorical(df_example)
```

### Discretize Continuous Variables
We will also want to discretize some of our continuous features.
```python
%psource cleaner.discretize
```

```python
bin_columns = [
    'students_reached',
    'total_price_including_optional_support'
]
df_example, _ = cleaner.discretize(df_example, bin_columns)
```

### Label Data
Finally, we will need to label our data. We will also drop the
'datefullyfunded' column once we've used it to generate our data.

```python
%psource cleaner.label
```
```python
label_colname = 'not_funded_in_60_days'
df_example = cleaner.label(df_example, label_colname)
```

The example cleaned dataset now looks like the following:
```python
df_example.head()
```

We see below that 29% of projects are not funded within 60 days of being
posted.
```python
df_example.not_funded_in_60_days.value_counts(normalize=True)
```

We can use this value later as our default threshold.
```python
default_threshold = 29
```

## Parameter Selection
### Training Data
We separate our data into a validation and a test set. The validation set will
be used for experimenting with model parameters. The test set will be used to
evaluate our parameterized models at different thresholds.

```python
begin = df.date_posted.min()
end = df.date_posted.max()
splits = pipeline.time_split(df, 'date_posted', begin, end, 4)
[(df_train.date_posted.min(), df_train.date_posted.max(), df_test.date_posted.max()) \
    for df_train, df_test in splits]
```

Now that we have our splits, we clean all of our datasets:
```python
# Now clean
cleaned_splits = []
for df_train, df_test in splits:
    df_train_clean, domains, binner = \
        cleaner.clean(df_train, bin_columns, label_colname)
    df_test_clean, _, _ = \
        cleaner.clean(df_test, bin_columns, label_colname, domains, binner)
    cleaned_splits.append((df_train_clean, df_test_clean))

splits = cleaned_splits
```

We take only the first two splits to use for validation, leaving a holdout for
final evaluation later.
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
trainer = Trainer(*dfs_train, label_colname=label_colname, seed=SEED)
tester = Tester(*dfs_test, label_colname=label_colname)
```

#### Logistic Regression
For our logistic regression model, we have to decide on a value for $c$:

```python
lr_results = ResultCollection()

for c in [1, 10, 100]:
    models = trainer.logistic_regression(c=c)
    lr_results.join('c_' + str(c), tester.test(*models,
                                               threshold=default_threshold))
```
```python
lr_results.statistic('auc')
```

It looks like using $c = 10$ is the best here.

#### Decision Tree
For our decision tree model, we need to experiment with tree depth to see what
depth is best.

```python
tree_results = ResultCollection()

tree_no_max_models = trainer.decision_tree(max_depth=None)
tree_results.join('no_max', tester.test(*tree_no_max_models,
                                        threshold=default_threshold))

for max_depth in [6, 12, 24]:
    models = trainer.decision_tree(max_depth=max_depth)
    tree_results.join(str(max_depth) + '_max', tester.test(*models,
                                                           threshold=default_threshold))

tree_results.statistic('auc')
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
    k_nearest_results.join('k_' + str(k), tester.test(*models,
                                                      threshold=default_threshold))

k_nearest_results.statistic('auc')
```

It looks like we want to set $k$ to around 12.

#### SVM
For our support vector machine model, we will want to experiment with setting
$c$, which is a penalty term that should be lower when we have noisy data.

```python
svm_results = ResultCollection()

for c in [1, 10, 100]:
    models = trainer.linear_svm(c=c)
    svm_results.join('c_' + str(c), tester.test(*models,
                                                threshold=default_threshold))

svm_results.statistic('auc')
```

It looks like choosing $c$ doesn't make much of a difference, so we might as
well go with the default of $c = 1$.

#### Random Forest
For random forests, we need to decide on the number of trees to use.

```python
forest_results = ResultCollection()

for n_trees in [100, 500, 1200]:
    models = trainer.forest(n_trees=n_trees)
    forest_results.join('n_' + str(n_trees), tester.test(*models,
                                                         threshold=default_threshold))

forest_results.statistic('auc')
```

Around 500 trees seems to be the best option.

#### Bagging
For the bagging approach, we need to decide on the number of base estimators to
create, or alternatively the number of bootstrap samples to draw.

```python
bagging_results = ResultCollection()

for n_estimators in [100, 500, 1200]:
    models = trainer.bagging(n_estimators=n_estimators)
    bagging_results.join('n_' + str(n_estimators), tester.test(*models,
                                                               threshold=default_threshold))

bagging_results.statistic('auc')
```

Using 100 estimators seems about right.

#### Boosting
For the boosting approach, we again need to decide on the number of base
estimators.

```python
boosting_results = ResultCollection()

for n_estimators in [100, 500, 1200]:
    models = trainer.boosting(n_estimators=n_estimators)
    boosting_results.join('n_' + str(n_estimators), tester.test(*models,
                                                                threshold=default_threshold))

boosting_results.statistic('auc')
```

Using 1200 estimators seems to be the best, most consistent option.

## Evaluation
Now that we are evaluating our models, we will want to use our final holdout
time split.

```python
df_train, df_test = holdout_split
trainer = Trainer(df_train, label_colname=label_colname, seed=SEED)
tester = Tester(df_test, label_colname=label_colname)
```

Next we set all of our model parameters according to the analysis we did in the
previous section and then train all of our models on our training set. _**We do
not pick new parameters here**, because the whole point of picking the
parameters without using the holdout set is to avoid overfitting to the holdout
data._ (See [this
guide](https://scikit-learn.org/stable/modules/cross_validation.html) from the
Scikitlearn documentation.) We then test our trained models on our holdout test
set.

```python
params = {
    'linear_regression': { 'c': 10 },
    'decision_tree': { 'max_depth': None },
    'k_nearest': { 'k': 12 },
    'svm': { 'c': 1 },
    'forest': { 'n_trees': 500 },
    'bagging': { 'n_estimators': 100 },
    'boosting': { 'n_estimators': 1200 }
}

models = trainer.train_all(parameters=params)
results = tester.evaluate(models)
results.df
```
```python
results.plot_statistic('auc', xlabel='model', ylim=(0.4, 0.6))
```

The above graph shows the AUC performance of our models with no threshold set.
This graph suggests that using either a linear SVM or logistic regression
model is the right approach.

We also want to test using our thresholds:
```python
threshold_results = \
    tester.evaluate(models, thresholds=[1, 2, 5, 10, 20, 30, 40, 50])
threshold_results.df
```
```python
threshold_results.plot_statistic('auc',
                                 xlabel='threshold')
```

This graph shows that the linear SVM and logistic regression models are the
best models at many thresholds, though the random forest model does well also.

We might also want to look at how precision and recall vary over our
thresholds:
```python
threshold_results.plot_statistic('precision',
                                 xlabel='threshold')
```
```python
threshold_results.plot_statistic('recall',
                                 xlabel='threshold')
```

If we were targeting a specific threshold, then we would probably want to
maximize precision. In that case, the above graph showing precision would be
important. If we had only a small amount of resources available to intervene,
then we might consider the random forest model, even though the linear SVM and
logistic regression models do better at higher thresholds.

Here are combined precision / recall curves for the linear SVM model and the
random forest model:
```python
threshold_results.plot_precision_recall('linear_svm')
```
```python
threshold_results.plot_precision_recall('forest')
```

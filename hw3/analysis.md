---
jupyter:
  kernelspec:
    name: python3
    langauge: python
    display_name: Python 3
---

# Predicting Funding for Projects on DonorsChoose.org
## Goal
Predict whether a project posted on DonorsChoose.org will get funded within 60
days of first being posted on the site.

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

## Model Building
### Training Data
We separate our data into a validation and a test set. The validation set will
be used for experimenting with model parameters. The test set will be used to
evaluate our parameterized models at different thresholds.

```python
splits = pipeline.time_split(df, 3)
[(df_train.index.min(), df_train.index.max(), df_test.index.max()) \
    for df_train, df_test in splits]
```
```python
validation_splits = splits[:-1]
test_split = splits[-1:]
```

### Models
We will use our validation set to parameterize our models. This will give us
two validation training sets.
```python
dfs_train = [df_train for df_train, _ in validation_splits]
dfs_test = [df_test for _, df_test in validation_splits]
[('split' + str(i), df_train.index.min(), df_train.index.max()) \
    for i, df_train in enumerate(dfs_train, 1)]
```
```python
[('split' + str(i), df_test.index.min(), df_test.index.max()) \
    for i, df_test in enumerate(dfs_test, 1)]
```

#### Logistic Regression
For our logistic regression model, we have to decide whether to use L1 or L2
regularization.

```python
from pipeline import ResultCollection

lr_results = ResultCollection()
lr_l1_models = pipeline.logistic_regression(dfs_train, label_colname,
                                            penalty='l1')
lr_results.add('L1', pipeline.test_all(lr_l1_models, dfs_test, label_colname))

lr_l2_models = pipeline.logistic_regression(dfs_train, label_colname,
                                            penalty='l2')
lr_results.add('L2', pipeline.test_all(lr_l2_models, dfs_test, label_colname))
lr_results.df
```
```python
lr_results.plot_statistic('f1')
```

It looks like using L1 regression here is marginally better, though both models
perform poorly.

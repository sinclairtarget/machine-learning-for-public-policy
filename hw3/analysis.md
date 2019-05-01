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
df.school_charter = df.school_charter == 't'
df.school_magnet = df.school_magnet == 't'
df.eligible_double_your_impact_match = \
    df.eligible_double_your_impact_match == 't'
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
df['not_funded_in_60_days'] = \
    (df.datefullyfunded - df.date_posted).apply(lambda d: d.days > 60)
```

We drop our date columns now that we have our labels:
```python
df = df.drop(columns=['date_posted',
                      'datefullyfunded'])
```

Our final dataframe looks like the following:
```python
df.head()
```

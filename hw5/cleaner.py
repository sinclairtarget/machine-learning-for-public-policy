"""
This file contains cleaning logic to use when loading the dataset.

See the analysis for more information about each function.
"""
import pipeline
import pandas as pd

def unnecessary_columns(df):
    return df.drop(columns=['projectid',
                            'teacher_acctid',
                            'schoolid',
                            'school_ncesid',
                            'school_city',
                            'school_district',
                            'school_county'])


def fix_types(df):
    for colname in ['school_charter',
                    'school_magnet',
                    'eligible_double_your_impact_match']:
        df[colname] = (df[colname] == 't').astype(float)

    return df


def handle_missing(df):
    df = df.drop(columns=['secondary_focus_subject',
                          'secondary_focus_area'])

    # Impute school_metro
    pipeline.impute(df, 'school_metro', how='mode')

    df = df.dropna()
    return df


def handle_categorical(df, domains=None):
    categorical_columns = pipeline.categorical_columns(df)

    if domains:
        pipeline.dummify_domain(df, domains, *categorical_columns)
    else:
        domains = pipeline.dummify(df, *categorical_columns)

    df = df.drop(columns=categorical_columns)
    return df, domains


def discretize(df, columns, binner=None):
    if not binner:
        binner = pipeline.Binner(n_bins=4, colnames=columns)
        binner.fit(df)

    df = binner.transform(df)
    return df.drop(columns=columns), binner


def label(df, label_colname):
    df[label_colname] = (
        (df.datefullyfunded - df.date_posted)
        .apply(lambda d: d.days > 60)
        .astype(float)
    )

    df = df.drop(columns=['datefullyfunded', 'date_posted'])
    return df


def clean(df, bin_columns, label_colname, domains=None, binner=None):
    df = unnecessary_columns(df)
    df = fix_types(df)
    df = handle_missing(df)
    df, domains = handle_categorical(df, domains)
    df, binner = discretize(df, bin_columns, binner)
    df = label(df, label_colname)
    return df, domains, binner

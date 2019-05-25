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
    df = df.dropna()
    return df


def handle_categorical(df):
    categorical_columns = pipeline.categorical_columns(df)
    pipeline.dummify(df, *categorical_columns)
    df = df.drop(columns=categorical_columns)
    return df


def discretize(df, binner=None):
    columns = [
        'students_reached',
        'total_price_including_optional_support'
    ]

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

    df = df.drop(columns=['datefullyfunded'])
    return df


def clean(df):
    df = unnecessary_columns(df)
    df = fix_types(df)
    df = handle_missing(df)
    df = handle_categorical(df)
    df, _ = discretize(df)
    df = label(df)
    return df

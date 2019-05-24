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
                            'school_state',
                            'school_metro',
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


def label(df, label_colname):
    df[label_colname] = (
        (df.datefullyfunded - df.date_posted)
        .apply(lambda d: d.days > 60)
        .astype(float)
    )

    df = df.sort_values('date_posted')
    df = df.reset_index(drop=True)

    df = df.drop(columns=['date_posted',
                          'datefullyfunded'])
    return df


def clean(df):
    df = unnecessary_columns(df)
    df = fix_types(df)
    df = handle_missing(df)
    df = handle_categorical(df)
    df = label(df)
    return df

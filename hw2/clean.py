import pipeline

df = pipeline.read_csv('credit-data.csv')

pipeline.impute(df, 'MonthlyIncome')
pipeline.impute(df, 'NumberOfDependents')

pipeline.bin(df, 'MonthlyIncome', [
    0,
    20000,
    40000,
    60000,
    80000,
    100000,
    120000,
    140000,
    160000,
    180000,
    200000
], labels = [
    '<20000',
    '<40000',
    '<60000',
    '<80000',
    '<100000',
    '<120000',
    '<140000',
    '<160000',
    '<180000',
    '<200000',
    '>200000'
])

pipeline.dummify(df, 'MonthlyIncome_binned')
pipeline.dummify(df, 'zipcode')

df = df.drop(columns=['MonthlyIncome'])
df = df.drop(columns=['MonthlyIncome_binned'])
df = df.drop(columns=['zipcode'])

pipeline.write_csv(df, 'credit-data-cleaned.csv')

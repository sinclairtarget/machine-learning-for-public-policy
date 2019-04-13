import pipeline

df = pipeline.read_csv('credit-data-cleaned.csv')

# Presumably this ID isn't helpful in prediction
df = df.drop(columns=['PersonID'])

pipeline.logistic_regression(df, 'SeriousDlqin2yrs')
pipeline.write_csv(df, 'credit-data-predicted.csv')

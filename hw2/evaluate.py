import pipeline

df = pipeline.read_csv('credit-data-predicted.csv')
accuracy = round(pipeline.accuracy(df, 'SeriousDlqin2yrs'), 2)

print(f"Model predicted {accuracy * 100}% of target values correctly.")

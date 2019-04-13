import pipeline
from sklearn.linear_model import LogisticRegression

df = pipeline.read_csv('credit-data-cleaned.csv')

# Presumably this ID isn't helpful in prediction
df = df.drop(columns=['PersonID'])

y = df['SeriousDlqin2yrs'].values
X = df.drop(columns=['SeriousDlqin2yrs']).values

model = LogisticRegression(solver='liblinear')
model.fit(X, y)

df['SeriousDlqin2yrs_predict'] = model.predict(X)

pipeline.write_csv(df, 'credit-data-predicted.csv')

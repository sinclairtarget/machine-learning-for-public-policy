from sklearn.linear_model import LogisticRegression

def logistic_regression(df, target_col, predict_col=None):
    if predict_col == None:
        predict_col = target_col + '_predict'

    y = df[target_col].values
    X = df.drop(columns=[target_col]).values

    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)

    df[predict_col] = model.predict(X)

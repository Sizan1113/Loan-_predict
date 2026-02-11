# Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data Load
df = pd.read_csv('../../financial_loan_data.csv')


def model_train():

    X_var = 'annual_income'
    Y_var = 'loan_amount'

    # feature
    X = df[[X_var]]
    # target
    Y = df[Y_var]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    return X_var, Y_var, X, Y, model, Y_pred

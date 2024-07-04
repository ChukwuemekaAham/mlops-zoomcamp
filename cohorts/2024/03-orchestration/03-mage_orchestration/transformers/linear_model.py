from pandas import DataFrame
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def linear_modeling(df: DataFrame):

    categorical = ["PULocationID", "DOLocationID"]
    print("Select categorical variables")
    final_df = df[categorical]
    print(final_df.head())
    print("transform df to dict")
    train_dicts = final_df.to_dict(orient='records')
    print(train_dicts[:10])
    print("Fit and transform a dict vectorizer from the data")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    print("Select target from initial df")
    y_train = df["duration"]
    print(y_train[:10])

    print("Fit a linear regression model with default parameters")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("Predict from training dataset")
    y_pred = lr.predict(X_train)
    print(y_pred[:10])

    print("Calculate mean squared error on training dataset")
    error = mean_squared_error(y_pred, y_train, squared=False)
    print(error)

    print("Determine intercept")
    print(lr.intercept_)

    return dv, lr
    
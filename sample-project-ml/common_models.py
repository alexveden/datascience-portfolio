from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def calc_cross_val_scores(model, X, y, display=True):
    """
    Displays model scores based on cross-validation method
    """
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)

    if display:
        print(rmse_scores)
        print(f'Mean: {np.mean(rmse_scores)}')
        print(f'StDev: {np.std(rmse_scores)}')

    return np.mean(rmse_scores)


def calc_score(model, X_train, y_train, X_test, y_test, display=True):
    """
    Trains model on X_train dataset and displays its score on X_test dataset
    """
    model.fit(X_train, y_train)

    predicted_y = model.predict(X_test)

    rmse_score = np.sqrt(mean_squared_error(y_test, predicted_y))

    if display:
        print(f"RMSE Score for test dataset: {rmse_score}")

    return rmse_score


def model_base():
    """
    Base model with will be used for results comparison and progress tracking
    """
    model = RandomForestRegressor(random_state=64, n_estimators=100)
    return model
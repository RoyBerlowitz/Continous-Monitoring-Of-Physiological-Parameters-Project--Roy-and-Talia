from .logistic_regression_model import train_logistic_regression
from .xgboost_model import train_xgboost

def train_model(X_selected,Y_train):
    return train_logistic_regression(X_selected,Y_train), train_xgboost(X_selected,Y_train)
    return 'Model_A','Model_B'
import copy

from .logistic_regression_model import train_logistic_regression, find_best_hp_logistic_regression
from .xgboost_model import train_xgboost, find_best_hp_xgboost
from .SVM_classifier import find_best_SVM_parameters, train_SVM
from .Random_forest_model import find_best_random_forrest_parameters, train_random_forest_classifier
from .consts import ModelNames
#from .Random_forest_model import

def choose_hyperparameters(train_df, target, model=ModelNames.SVM, n_jobs = -1, n_iterations = 50, split_name = "Individual Split"):

    df_for_hyperparameters = copy.deepcopy(train_df)
    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    # we don't want the administrative features to be a part of the model, so we remove them from the hyperparameteres loop.
    df_for_hyperparameters = df_for_hyperparameters[[col for col in df_for_hyperparameters.columns if col not in administrative_features]]

    if model == ModelNames.SVM:
        best_SVM_parameters = find_best_SVM_parameters(df_for_hyperparameters, target, n_jobs, n_iterations = n_iterations, split_name = split_name)
        return best_SVM_parameters

    if model == ModelNames.RANDOM_FOREST:
        best_Random_Forrest_parameters = find_best_random_forrest_parameters(df_for_hyperparameters, target, n_jobs, n_iterations = n_iterations, split_name = split_name)
        return best_Random_Forrest_parameters

    if model == ModelNames.LOGISTIC:
        return find_best_hp_logistic_regression(df_for_hyperparameters, target, split_name = split_name)

    if model == ModelNames.XGBOOST:
        return find_best_hp_xgboost(df_for_hyperparameters, target, split_name = split_name)

    return None

def train_model(X_selected,Y_train, best_parameters, model_name):

    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    train_x = copy.deepcopy(X_selected)
    # train_x = train_x.drop(columns = administrative_features, inplace = True)
    train_x = train_x[[col for col in train_x.columns if col not in administrative_features]]

    if model_name == ModelNames.SVM:
        SVM_model = train_SVM( train_x, Y_train, best_parameters, name = "Individual Split")
        return SVM_model
    if model_name == ModelNames.RANDOM_FOREST:
        random_forest_model = train_random_forest_classifier( train_x, Y_train,best_parameters, name = "Individual Split")
        return random_forest_model
    if model_name == ModelNames.LOGISTIC:
        return train_logistic_regression( train_x, Y_train,best_parameters)
    if model_name == ModelNames.XGBOOST:
        return train_xgboost(train_x, Y_train, best_parameters)

    return None
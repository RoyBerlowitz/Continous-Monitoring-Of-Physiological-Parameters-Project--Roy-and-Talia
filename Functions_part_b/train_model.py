import copy

from .logistic_regression_model import train_logistic_regression
from .xgboost_model import train_xgboost
from .SVM_classifier import find_best_SVM_parameters, train_SVM
from .Random_forest_model import find_best_random_forrest_parameters, train_random_forest_classifier
#from .Random_forest_model import

def choose_hyperparameters(train_df, target, n_dimensions,   params_range: dict, model="SVM",n_jobs = -1, n_iterations = 50, split_name = "Individual Split"):
    df_for_hyperparameters = copy.deepcopy(train_df)
    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    # we don't want the administrative features to be a part of the model, so we remove them from the hyperparameteres loop.
    df_for_hyperparameters = df_for_hyperparameters.drop(columns = administrative_features, inplace = True)

    if model == "SVM":
        best_SVM_parameters = find_best_SVM_parameters(df_for_hyperparameters, target, n_jobs, n_iterations = n_iterations, split_name = split_name)
        return best_SVM_parameters

    if model == "Random Forest":
        best_Random_Forrest_parameters = find_best_random_forrest_parameters(df_for_hyperparameters, target, n_jobs, n_iterations = n_iterations, split_name = split_name)
        return best_Random_Forrest_parameters

    # if model == "Logistic Regression":
    #     best_logistic_regression_parameters = find_best_logistic_regression_parameters(df_for_hyperparameters, target, n_jobs, n_iterations = n_iterations, split_name = split_name)
    #     return best_logistic_regression_parameters
    #
    # if model == "XGBoost":
    #     best_XGBoost_parameters = find_best_XGBoost_parameters(df_for_hyperparameters, target, n_jobs, n_iterations = n_iterations, split_name = split_name)
    #     return best_XGBoost_parameters



def train_model(X_selected,Y_train, best_parameters, model_name):
    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    train_x = copy.deepcopy(X_selected)
    train_x = train_x.drop(columns = administrative_features, inplace = True)
    if model_name == "SVM":
        SVM_model = train_SVM( train_x, Y_train,val_df, val_labels, best_parameters, name = "Individual Split")
        return SVM_model
    if model_name == "Random Forest":
        random_forest_model = train_random_forest_classifier( train_x, Y_train,best_parameters, name = "Individual Split")


    return train_logistic_regression(X_selected,Y_train), train_xgboost(X_selected,Y_train)

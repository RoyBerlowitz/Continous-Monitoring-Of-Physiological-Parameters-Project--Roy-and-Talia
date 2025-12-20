import copy

from .logistic_regression_model import train_logistic_regression, find_best_hp_logistic_regression
from .xgboost_model import train_xgboost, find_best_hp_xgboost
from .SVM_classifier import find_best_SVM_parameters, train_SVM
from .Random_forest_model import find_best_random_forrest_parameters, train_random_forest_classifier
from .consts import ModelNames
#from .Random_forest_model import

def choose_hyperparameters(train_df, labels, model=ModelNames.SVM, n_jobs = -1, n_iterations = 50, split_name = "Individual Split", split_by_group_flag = False, wrapper_text = ""):
    # This function is meant to be the Super function for finding the best Hyper-parameters.
    # by defining the model name, we choose on which model we should conduct the search for hyperparameters.
    # split_by_group is meant to ensure we leave (1/number of folds) of the groups out when we work on the group split - to resemble the case

    # we create a copy to avoid changing the original data frame
    df_for_hyperparameters = copy.deepcopy(train_df)
    target = copy.deepcopy(labels)
    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    group_indicator = df_for_hyperparameters['Group number']
    # we don't want the administrative features to be a part of the model, so we remove them from the hyperparameteres loop.
    df_for_hyperparameters = df_for_hyperparameters[[col for col in df_for_hyperparameters.columns if col not in administrative_features]]

    # we call every model's hyperparameter function. the evaluation method is the PRC as we want to maximize both precision and sensitivity.
    if model == ModelNames.SVM:
        best_SVM_parameters = find_best_SVM_parameters(df_for_hyperparameters, target,group_indicator, n_jobs, n_iterations = n_iterations, split_name = split_name, split_by_group_flag = split_by_group_flag, wrapper_text = wrapper_text)
        return best_SVM_parameters

    if model == ModelNames.RANDOM_FOREST:
        best_Random_Forrest_parameters = find_best_random_forrest_parameters(df_for_hyperparameters, target,group_indicator, n_jobs, n_iterations = n_iterations, split_name = split_name, split_by_group_flag = split_by_group_flag, wrapper_text=wrapper_text)
        return best_Random_Forrest_parameters

    if model == ModelNames.LOGISTIC:
        return find_best_hp_logistic_regression(df_for_hyperparameters, target, split_name, split_by_group_flag, group_indicator, wrapper_text)

    if model == ModelNames.XGBOOST:
        return find_best_hp_xgboost(df_for_hyperparameters, target, split_name, split_by_group_flag, group_indicator, wrapper_text)

    return None

def train_model(X_selected,y_train, best_parameters, model_name, split_by_group_flag = False):

    # split_by_group is meant to ensure we leave (1/number of folds) of the groups out when we work on the group split - to resemble that case.
    # it is relevant in all models but random forrest
    # we create a copy to avoid changing the original data frame
    train_x = copy.deepcopy(X_selected)
    Y_train = copy.deepcopy(y_train)

    group_indicator=None
    if split_by_group_flag:
        group_indicator = X_selected['Group number']

    # we don't want the administrative features to be a part of the model, so we remove them from the hyperparameteres loop.
    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    train_x = train_x[[col for col in train_x.columns if col not in administrative_features]]

    # we call every model's train function. at the end we save the optimal threshold in the ROC curve
    # and the threshold which offers the best F1 in the PRC curve.
    if model_name == ModelNames.SVM:
        SVM_model = train_SVM( train_x, Y_train, best_parameters, name = "Individual Split", split_by_group_flag = split_by_group_flag)
        return SVM_model
    if model_name == ModelNames.RANDOM_FOREST:
        # the finding of threshold is based on the OOB data so no need for k-folds and group flag
        random_forest_model = train_random_forest_classifier( train_x, Y_train,best_parameters, name = "Individual Split")
        return random_forest_model
    if model_name == ModelNames.LOGISTIC:
        return train_logistic_regression( train_x, Y_train,best_parameters, split_by_group_flag=split_by_group_flag, group_indicator=group_indicator)
    if model_name == ModelNames.XGBOOST:
        return train_xgboost(train_x, Y_train, best_parameters, split_by_group_flag=split_by_group_flag, group_indicator=group_indicator)

    return None
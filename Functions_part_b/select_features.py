from .wrapper_feature_selection import select_features_wrapper
from .filter_feature_selection import  select_features_filter
from .consts import ModelNames

def select_features(X, Y, function_variable: list, split_name="Individual_split",  selection_flag = "wrapper", split_by_group_flag = False):
    # This function is meant to be the Super function for the feature selection.
    # by setting the selection flag, we choose between filter and wrapper feature selection
    # # the function variables is a list of the variables needed for each feature selection function method.
    if selection_flag == "filter":
        #Here the only needed variable is the stopping criteria for the MRMR
        [stopping_criteria] = function_variable
        selected_features = select_features_filter(X, Y, split_name=split_name, stopping_criteria=0)
    elif selection_flag == "wrapper":
        # Here we need the hyper-parameters for the model, the range of features to choose from and the model type.
        # we evaluate the model by PRC
        [frozen_params,n_features_range,model_type] = function_variable
        selected_features = select_features_wrapper(X, Y, frozen_params,
                                        n_features_range=n_features_range,
                                        model_type=model_type, split_name=split_name, split_by_group_flag = split_by_group_flag)
    return selected_features
import pickle
import numpy as np
import time

from Functions_part_b import (load_cache_or_compute, select_features, choose_hyperparameters, train_model,
                              evaluate_model, wrapper_feature_selection)
from Functions_part_b.select_features import select_features
from consts import (ModelNames, chosen_hp_split1, chosen_hp_split2)

# Here are some definition for the part b running.
# the administrative features are the feature which hold the details of every recording. they are important to keep, but the model should not train on them
admin_features = ['First second of the activity','Last second of the activity','Participant ID','Group number','Recording number','Protocol']
# As explained in the wrapper_feature_selection file, we first implemented a filter feature selection with CACC discertization, and MRMR algorithm for the selection.
# After examining 4 models (SVM, Logistic Regression, Gradient Boosting and Random Forest), we came to realization that Gradient Bossting and Random forrest are the better operating models.
# Then, we decide to re-select feature based on those model by wrapper selection method.
# because the complexity and run time of fitting the best hyperparameters for each candiate number of features, we took the best hyper-parameters from the run on the filter selected features,
# and use them for the model wrapper feature selection process (more explained in the function files)
wrapper_params_split_1 = {
    ModelNames.RANDOM_FOREST:{'Random_Forest__class_weight': 'balanced_subsample', 'Random_Forest__max_depth': 26, 'Random_Forest__max_samples': np.float64(0.6035119926400068), 'Random_Forest__min_samples_leaf': 13, 'Random_Forest__min_samples_split': 21, 'Random_Forest__n_estimators': 489},
    ModelNames.XGBOOST:{'colsample_bytree': np.float64(0.6161734358153725), 'gamma': np.float64(0.21319886690573622), 'learning_rate': np.float64(0.014581398811193679), 'max_depth': 6, 'n_estimators': 254, 'scale_pos_weight': 1, 'subsample': np.float64(0.6125716742746937)},
}
wrapper_params_split_2 = {
    ModelNames.RANDOM_FOREST:{'Random_Forest__class_weight': 'balanced', 'Random_Forest__max_depth': 24, 'Random_Forest__max_samples': np.float64(0.7852979148891981), 'Random_Forest__min_samples_leaf': 10, 'Random_Forest__min_samples_split': 24, 'Random_Forest__n_estimators': 317},
    ModelNames.XGBOOST:{'colsample_bytree': np.float64(0.6161734358153725), 'gamma': np.float64(0.21319886690573622), 'learning_rate': np.float64(0.014581398811193679), 'max_depth': 6, 'n_estimators': 254, 'scale_pos_weight': 1, 'subsample': np.float64(0.6125716742746937)},
}
# Those are the examined number of features for the wrappers
n_features_range = [3, 5, 7, 10, 12, 15, 17, 19, 20]
# we define the model the wrapper should run on.
wrapper_models = [ModelNames.XGBOOST, ModelNames.RANDOM_FOREST]
wrapper_models = [ModelNames.RANDOM_FOREST]
# wrapper_models = [ModelNames.XGBOOST]
# the wrapper parameters
chosen_hp_split1 = {ModelNames.RANDOM_FOREST: [wrapper_params_split_1[ModelNames.RANDOM_FOREST], n_features_range, ModelNames.RANDOM_FOREST], ModelNames.XGBOOST: [wrapper_params_split_1[ModelNames.XGBOOST], n_features_range, ModelNames.XGBOOST]}
chosen_hp_split2 = {ModelNames.RANDOM_FOREST: [wrapper_params_split_2[ModelNames.RANDOM_FOREST], n_features_range, ModelNames.RANDOM_FOREST], ModelNames.XGBOOST: [wrapper_params_split_2[ModelNames.XGBOOST], n_features_range, ModelNames.XGBOOST]}




def run_part_b_specific_dataset(X_train, X_test, y_train, y_test, scaler, models_to_run, split_name, split_by_group_flag=False, use_wrapper=False, chosen_hp=None, save_cache=False, force_recompute_select_features=True, force_recompute_find_hp=True, force_recompute_train_model=True, force_recompute_evaluate_model=True, subsampling_flg = False):

    features = [col for col in X_train.columns if col not in admin_features]
    wrapper_text = "" if not use_wrapper else "_wrapper"

    print(f'\033[34mStarting on split {split_name}{wrapper_text} ==========================================\033[0m')

    ##--------------- Part A: select features ----------------##
    # if use_wrapper flag is False, we run the filter feature selection, which includes MRMR method with CACC discertization.
    # as we decided to use wrapper, we submitted the mission with the flag as a false.
    # in this case the chosen hyper-parameters is the stopping rule for the MRMR, which is zero (more details in select_features_filter and MRMR_features_selection file)
    if not use_wrapper:
        selected_feats = load_cache_or_compute(
            f"{split_name}_select_features.pkl",
            lambda: select_features(X_train, y_train, chosen_hp, selection_flag = "filter", split_name=split_name),
            force_recompute=force_recompute_select_features,
            save=save_cache
        )

        print('\033[32mFeature selection completed\033[0m')

    ##--------------- Part B: train model -----------##
    trained_models = {}

    for model_name in models_to_run:
        # if we selected to use wrapper, the flag will be true and we commit the selection for each model seperately.
        if use_wrapper:
            selected_feats = load_cache_or_compute(
                    f"{split_name}_{model_name}_wrapper_select_features.pkl",
                    lambda: select_features(X_train, y_train,  chosen_hp[model_name], split_name=split_name+model_name, selection_flag = "wrapper", split_by_group_flag = split_by_group_flag),
                    force_recompute=force_recompute_select_features,
                    save=save_cache
                )
            print('\033[32mFeature selection completed\033[0m')
        # we update the select features
        X_selected = X_train[selected_feats+admin_features]
        X_test = X_test[selected_feats+admin_features]

        #get best hyperparameters
        model_best_hp = load_cache_or_compute(
            f"{split_name}_{model_name}{wrapper_text}_best_hp.pkl",
            lambda: choose_hyperparameters(X_selected,y_train,model_name,split_name=split_name+model_name,split_by_group_flag=split_by_group_flag,wrapper_text=wrapper_text, subsampling_flg = subsampling_flg),
            force_recompute=force_recompute_find_hp,
            save=save_cache
        )

        #train chosen model
        trained_models[model_name],df_for_time_classification = load_cache_or_compute(
            f"{split_name}_{model_name}{wrapper_text}_trained_model.pkl",
            lambda: train_model(X_selected, y_train, model_best_hp, model_name,split_by_group_flag=split_by_group_flag, subsampling_flg = subsampling_flg),
            force_recompute=force_recompute_train_model,
            save=save_cache
        )

        print(f'\033[32mFinished training {model_name} {split_name}{wrapper_text}\033[0m')

    print('\033[32mTrain model completed\033[0m')

    # --------------- Part C: evaluate model ---------------##
    # Here, we evaluate the model.
    stat_values =  load_cache_or_compute(
        f"{split_name}{wrapper_text}_evaluate_models.pkl",
        lambda: evaluate_model(list(trained_models.values()), list(trained_models.keys()),
                               X_test[selected_feats+admin_features], y_test,
                               split_name = split_name,
                               save_model_outputs=True, wrapper_text=wrapper_text),
        force_recompute=force_recompute_evaluate_model,
        save=save_cache
    )

    print('\033[32mEvaluate model completed\033[0m')

    return  stat_values

def run_part_b(chosen_hp_split1=None, chosen_hp_split2=None, wrapper_models = None, save_cache=False, force_recompute_select_features=True, force_recompute_find_hp=True, force_recompute_train_model=True, force_recompute_evaluate_model=True, use_wrapper=False, subsampling_flg = False):
    #here we preform the entire run_part_b
    models = [ModelNames.LOGISTIC, ModelNames.XGBOOST, ModelNames.SVM, ModelNames.RANDOM_FOREST]  #all

    #load part a
    part_a_res_cache_path = "part_a_final_output.pkl"
    with open(part_a_res_cache_path, "rb") as f:
        part_a_res = pickle.load(f)

    split1_vet_features, split2_vet_features = part_a_res

    # # #split1
    # split1_X_train, split1_X_test, split1_y_train, split1_y_test, scalers = split1_vet_features
    # if not use_wrapper:
    #     # running with filter selection
    #     run_part_b_specific_dataset(split1_X_train, split1_X_test, split1_y_train, split1_y_test, None, models, 'split_1',split_by_group_flag=False, use_wrapper=False,save_cache=save_cache,
    #                                 force_recompute_select_features=force_recompute_select_features, force_recompute_find_hp=force_recompute_find_hp,
    #                                 force_recompute_train_model=force_recompute_train_model, force_recompute_evaluate_model=force_recompute_evaluate_model,
    #                                 chosen_hp=[0])
    # else:
    #     #runing with wrapper selection
    #     run_part_b_specific_dataset(split1_X_train, split1_X_test, split1_y_train, split1_y_test, None, wrapper_models, 'split_1',split_by_group_flag=False,
    #                                 use_wrapper=True, save_cache=save_cache,
    #                                 force_recompute_select_features=force_recompute_select_features,
    #                                 force_recompute_find_hp=force_recompute_find_hp,
    #                                 force_recompute_train_model=force_recompute_train_model,
    #                                 force_recompute_evaluate_model=force_recompute_evaluate_model,
    #                                 chosen_hp=chosen_hp_split1)

    #split2
    X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler = split2_vet_features
    if not use_wrapper:
        # running with filter selection
        run_part_b_specific_dataset(X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler, models, 'split_2', split_by_group_flag=True, use_wrapper=False, save_cache=save_cache,
                                force_recompute_select_features=force_recompute_select_features, force_recompute_find_hp=force_recompute_find_hp,
                                force_recompute_train_model=force_recompute_train_model, force_recompute_evaluate_model=force_recompute_evaluate_model,
                                chosen_hp=[0], subsampling_flg = subsampling_flg)
    else:
        #runing with wrapper selection
        run_part_b_specific_dataset(X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler, wrapper_models, 'split_2', split_by_group_flag=True,
                                use_wrapper=True, save_cache=save_cache,
                                force_recompute_select_features=force_recompute_select_features,
                                force_recompute_find_hp=force_recompute_find_hp,
                                force_recompute_train_model=force_recompute_train_model,
                                force_recompute_evaluate_model=force_recompute_evaluate_model,subsampling_flg = subsampling_flg,
                                chosen_hp=chosen_hp_split2)

    return

start_time = time.time()
#run with wrapper to find best features - this is what we chose
run_part_b(chosen_hp_split1, chosen_hp_split2, wrapper_models, save_cache=True, force_recompute_select_features=True, force_recompute_find_hp=True,
                                force_recompute_train_model=True, force_recompute_evaluate_model=True, use_wrapper = True, subsampling_flg = False)
# run_part_b(chosen_hp_split1, chosen_hp_split2, wrapper_models, save_cache=False, force_recompute_select_features=False, force_recompute_find_hp=False,
#                                 force_recompute_train_model=True, force_recompute_evaluate_model=True, use_wrapper = True)
#run with filter to find best features
# run_part_b(chosen_hp_split1, chosen_hp_split2, wrapper_models, save_cache=True, force_recompute_select_features=True, force_recompute_find_hp=True,
#                                 force_recompute_train_model=True, force_recompute_evaluate_model=True, use_wrapper = False)
end_time = time.time()
print(f"Total time: {end_time - start_time} sec")

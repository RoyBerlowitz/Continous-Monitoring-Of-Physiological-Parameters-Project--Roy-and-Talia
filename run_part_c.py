import numpy as np
import pickle
import time
import os

from Functions_part_b import (select_features, choose_hyperparameters, train_model,
                              evaluate_model, ModelNames, chosen_hp_split1, chosen_hp_split2, wrapper_feature_selection)
from Functions_part_b.select_features import select_features
from Functions_part_c import (load_cache_or_compute, load_data, predict_times, create_test_time_df)

# the administrative features are the feature which hold the details of every recording. they are important to keep, but the model should not train on them
admin_features = ['First second of the activity','Last second of the activity','Participant ID','Group number','Recording number','Protocol']

wrapper_params_split_2 = {
    ModelNames.RANDOM_FOREST:{'Random_Forest__class_weight': 'balanced', 'Random_Forest__max_depth': 24, 'Random_Forest__max_samples': np.float64(0.7852979148891981), 'Random_Forest__min_samples_leaf': 10, 'Random_Forest__min_samples_split': 24, 'Random_Forest__n_estimators': 317},
    ModelNames.XGBOOST:{'colsample_bytree': np.float64(0.6161734358153725), 'gamma': np.float64(0.21319886690573622), 'learning_rate': np.float64(0.014581398811193679), 'max_depth': 6, 'n_estimators': 254, 'scale_pos_weight': 1, 'subsample': np.float64(0.6125716742746937)},
}
# Those are the examined number of features for the wrappers
n_features_range = [3, 5, 7, 10, 12, 15, 17, 19, 20]
# the wrapper parameters
chosen_hp = {ModelNames.RANDOM_FOREST: [wrapper_params_split_2[ModelNames.RANDOM_FOREST], n_features_range, ModelNames.RANDOM_FOREST], ModelNames.XGBOOST: [wrapper_params_split_2[ModelNames.XGBOOST], n_features_range, ModelNames.XGBOOST]}




def run_part_c(save_cache=False, force_recompute_load_data=True, force_recompute_handwashing_times=True, force_recompute_select_features=True, force_recompute_find_hp=True, force_recompute_train_model=True, force_recompute_evaluate_model=True):
    start_time = time.time()

    models = [ModelNames.XGBOOST, ModelNames.RANDOM_FOREST]
    models = [ModelNames.RANDOM_FOREST] #roee
    models = [ModelNames.XGBOOST] #talia

    split_name = 'split2'
    use_wrapper = True

    # load part a
    part_a_res_cache_path = "part_a_final_output.pkl"
    with open(part_a_res_cache_path, "rb") as f:
        part_a_res = pickle.load(f)

    split1_vet_features, split2_vet_features = part_a_res
    X_train, X_test, y_train, y_test, scaler = split2_vet_features

    ## ---------------- load data files ----------------
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_directory, "data")

    data_files = load_cache_or_compute(
        "load_data_output.pkl",
        lambda: load_data(data_path),
        force_recompute=force_recompute_load_data,
        save=save_cache
    )

    print('\033[32mData loaded\033[0m')

    features = [col for col in X_train.columns if col not in admin_features]
    wrapper_text = "" if not use_wrapper else "_wrapper"

    print(f'\033[34mStarting on split {split_name}{wrapper_text} ==========================================\033[0m')

    ##--------------- Part B: train model -----------##
    trained_models = {}
    train_time_dfs = {}

    for model_name in models:
        # if we selected to use wrapper, the flag will be true and we commit the selection for each model seperately.
        selected_feats = load_cache_or_compute(
                f"{model_name}_wrapper_select_features.pkl",
                lambda: select_features(X_train, y_train, chosen_hp[model_name], split_name=split_name+model_name, selection_flag = "wrapper", split_by_group_flag = True),
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
            lambda: choose_hyperparameters(X_selected,y_train,model_name,split_name=split_name+model_name,split_by_group_flag=True,wrapper_text=wrapper_text),
            force_recompute=force_recompute_find_hp,
            save=save_cache
        )

        #train chosen model
        trained_models[model_name], train_time_dfs[model_name] = load_cache_or_compute(
            f"{split_name}_{model_name}{wrapper_text}_trained_model.pkl",
            lambda: train_model(X_selected, y_train, model_best_hp, model_name,split_by_group_flag=True),
            force_recompute=force_recompute_train_model,
            save=save_cache
        )

        print(f'\033[32mFinished training {model_name} {split_name}{wrapper_text}\033[0m')

    print('\033[32mTrain model completed\033[0m')

    # --------------- Part C: evaluate model ---------------##
    # Create time_dfs for test
    test_time_dfs = {}
    for model_name in models:
        test_time_dfs[model_name] = load_cache_or_compute(
            f"{split_name}_{model_name}{wrapper_text}_test_time_df.pkl",
            lambda: create_test_time_df(X_test, trained_models[model_name], selected_feats),
            force_recompute=force_recompute_train_model,
            save=save_cache
        )
        predict_times(train_time_dfs[model_name], test_time_dfs[model_name], data_files, "No model")

    # # Here, we evaluate the model.
    # stat_values =  load_cache_or_compute(
    #     f"{split_name}{wrapper_text}_evaluate_models.pkl",
    #     lambda: evaluate_model(list(trained_models.values()), list(trained_models.keys()),
    #                            X_test[selected_feats+admin_features], y_test,
    #                            split_name = split_name,
    #                            save_model_outputs=True, wrapper_text=wrapper_text),
    #     force_recompute=force_recompute_evaluate_model,
    #     save=save_cache
    # )
    #
    # print('\033[32mEvaluate model completed\033[0m')

    end_time = time.time()
    print(f"Total time: {end_time - start_time} sec")

    # return  stat_values
    return

# ========================================================= Run =========================================================
if __name__ == "__main__":
    run_part_c(save_cache=True, force_recompute_load_data=False, force_recompute_handwashing_times=False, force_recompute_select_features=False, force_recompute_find_hp=False,force_recompute_train_model=False, force_recompute_evaluate_model=True)

import numpy as np
import pickle
import time
import os

# import numpy as np
# import torch
# import pandas as pd
#
# print("NumPy:", np.__version__)
# print("Torch:", torch.__version__)
# print("Pandas:", pd.__version__)
#
# a = np.array([1,2,3])
# t = torch.from_numpy(a)
# print(t)


import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from run_part_a import run_part_a
from Functions_part_b import (select_features, choose_hyperparameters, train_model,
                              evaluate_model, wrapper_feature_selection)
from Functions_part_b.select_features import select_features
from Functions_part_c import (load_cache_or_compute, load_data, predict_times, create_test_time_df)
from Functions.extract_features_helper_functions import get_cnn_embeddings
from consts import (ModelNames, ModelNamesSecondClassification, chosen_hp_split1, chosen_hp_split2)

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

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
data_path = os.path.join(script_directory, "data")
# split1_dfs, split2_dfs = run_part_a(data_path, save_cache=True, more_prints=True, force_recompute_load_data=False,
#                                     force_recompute_seg=False, force_recompute_features=True,
#                                     force_recompute_splits=True, force_recompute_vet_features=False)


def run_part_c(save_cache=False, force_recompute_load_data=True, force_recompute_select_features=True, force_recompute_find_hp=True, force_recompute_train_model=True, force_recompute_test_time_dfs=True, force_recompute_best_th = True, grp = None, force_recompute_evaluate_model=True):
    start_time = time.time()

    models = [ModelNames.XGBOOST, ModelNames.RANDOM_FOREST]
    models = [ModelNames.RANDOM_FOREST] #roee
    # models = [ModelNames.XGBOOST] #talia

    seconds_classification_models = [ModelNamesSecondClassification.NO_MODEL,ModelNamesSecondClassification.LOGISTIC, ModelNamesSecondClassification.MARKOV]
    #seconds_classification_models = [ModelNamesSecondClassification.LOGISTIC, ModelNamesSecondClassification.MARKOV]


    split_name = 'split2'
    use_wrapper = True

    # load part a
    part_a_res_cache_path = "splits.pkl"
    with open(part_a_res_cache_path, "rb") as f:
        part_a_res = pickle.load(f)

    split1_vet_features, split2_vet_features = part_a_res
    X_train, X_test, y_train, y_test = split2_vet_features

    #split test train again. remove all protocol
    resplit_train_test = True
    if resplit_train_test:
        X_all = pd.concat([X_train, X_test], axis=0)
        y_all = pd.concat([y_train, y_test], axis=0)
        X_all = X_all.sort_index()
        y_all = y_all.sort_index()

        mask = X_all['Protocol'] != 1
        X_all = X_all.loc[mask]
        y_all = y_all.loc[mask]

        # --- התיקון: שימוש בעמודה חדשה לפיצול במקום לדרוס את Group number ---
        X_all['Split_ID'] = X_all['Group number'] + X_all['Participant ID']
        print(f"Unique Split IDs: {X_all['Split_ID'].unique()}")

        # עדכון הפונקציה שתשתמש בעמודה החדשה כברירת מחדל או בקריאה
        def split_by_group_tuple(X, y, group_tuple, group_col='Split_ID'):  # שינינו את ברירת המחדל ל-Split_ID
            test_mask = X[group_col].isin(group_tuple)

            X_test = X.loc[test_mask]
            y_test = y.loc[test_mask]

            X_train = X.loc[~test_mask]
            y_train = y.loc[~test_mask]

            return X_train, X_test, y_train, y_test

        # שליחת העמודה החדשה לפונקציה
        X_train, X_test, y_train, y_test = split_by_group_tuple(X_all, y_all, [grp], group_col='Split_ID')

        # creating an embedder
        columns_names = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS',
                         'Mag_X-AXIS', 'Mag_Y-AXIS', 'Mag_Z-AXIS', 'Acc_SM', 'Mag_SM', 'Gyro_SM']

        columns_names_for_embedding = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS',
                                       'Gyro_Z-AXIS', ]
        X_train = get_cnn_embeddings(X_train,
                                     target=y_train,
                                     group_col="Group number",
                                     column_list=columns_names_for_embedding,
                                     test_flag=False,
                                     model_path='cnn_train_weights.pth',
                                     embedding_size=16,
                                     num_epochs=30,
                                     batch_size=64,
                                     dropout=0.25,
                                     steps=8)

        # getting rid of the columns with the vectors of values
        X_train = X_train.drop(labels=columns_names, axis=1)

        # activating the embedder on the test
        X_test = get_cnn_embeddings(X_test,
                                    target=y_test,
                                    group_col="Group number",
                                    column_list=columns_names_for_embedding,
                                    test_flag=True,
                                    model_path='cnn_train_weights.pth',
                                    embedding_size=16,
                                    num_epochs=2,
                                    batch_size=64)

        # getting rid of the columns with the vectors of values
        X_test = X_test.drop(labels=columns_names, axis=1)
        administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']

        informative_features =[
    "cnn_emb_7",
    "Acc_X-AXIS_acceleration_std",
    "cnn_emb_2",
    "Acc_X_Z_CORR",
    "cnn_emb_13",
    "cnn_emb_0",
    "Acc_Z-AXIS_CUSUM+_Feature",
    "cnn_emb_10",
    "Acc_Z-AXIS_CUSUM-_Feature",
    "Gyro_Z-AXIS_AbsCV",
    "Acc_SM_acceleration_median",
    "Gyro_SM_velocity_median",
    "Gyro_Y-AXIS_velocity_median",
    "Mag_Y-AXIS_median",
    "Acc_X-AXIS_velocity_skewness",
    "Mag_MEAN_AXES_CORR",
    "cnn_emb_6",
    "Gyro_X-AXIS_CUSUM-_Feature",
    "Gyro_SM_acceleration_kurtosis",
    "Acc_Z-AXIS_velocity_skewness"
]
        features_to_keep = administrative_features + informative_features
        X_train = X_train[features_to_keep]
        X_test = X_test[features_to_keep]
        # groups = X_all['Group number'] #'Participant ID',
        # gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
        # train_idx, test_idx = next(gss.split(X_all, y_all, groups))
        # X_train = X_all.iloc[train_idx]
        # X_test = X_all.iloc[test_idx]
        # y_train = y_all.iloc[train_idx]
        # y_test = y_all.iloc[test_idx]
        print(X_test['Group number'].unique())

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
                f"split2_{model_name}_wrapper_select_features.pkl",
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
            force_recompute=force_recompute_test_time_dfs,
            save=save_cache
        )

    # Choose thresholds and get evaluation metrics
    model_stats = {}
    for model_name in models:
        model_stats[model_name]={}
        for second_model in seconds_classification_models:
            model_stats[model_name][second_model] = load_cache_or_compute(
                f"{split_name}_{model_name}_{second_model}{wrapper_text}_choose_threshold_and_stas_per_second.pkl",
                lambda: predict_times(train_time_dfs[model_name], test_time_dfs[model_name], data_files, model_name,second_model),
                force_recompute=force_recompute_best_th,
                save=save_cache
            )

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
    return model_stats, X_test['Group number'].unique()

# ========================================================= Run =========================================================
if __name__ == "__main__":
    #run_part_c(save_cache=True)
    # run_part_c(save_cache=True,
    #            force_recompute_load_data=False,
    #            force_recompute_select_features=True,
    #            force_recompute_find_hp=True,
    #            force_recompute_train_model=True,
    #            force_recompute_test_time_dfs=True,
    #            force_recompute_best_th = True, )
    grps = ['15A','27A','31A','42A','58A','64A','79A','93A', '15B','27B','31B','42B','58B','64B','79B','93B']
    #grps = ['15','27','31','42','58','64','79','93',]

    # pairs = list(itertools.combinations(grps, 2))

    all_res = {}
    for gr in grps:
        print(f"gs is {gr}")
        res, gs = run_part_c(save_cache=True, force_recompute_load_data=False, force_recompute_select_features=False,
                             force_recompute_find_hp=False, force_recompute_train_model=True,
                             # force_recompute_find_hp=False, force_recompute_train_model=False,
                             force_recompute_evaluate_model=True, force_recompute_test_time_dfs=True, grp=gr)
        all_res[tuple(gs)] = res[ModelNames.RANDOM_FOREST]

    print(all_res)

    # all_res = {}
    # for i in range(10):
    #     res, gs = run_part_c(save_cache=True, force_recompute_load_data=False, force_recompute_select_features=False,
    #                          force_recompute_find_hp=True,force_recompute_train_model=True, force_recompute_evaluate_model=True, force_recompute_test_time_dfs=True)
    #     print(f"gs is {gs}")
    #     all_res[tuple(gs)] = res[ModelNames.XGBOOST]
    #
    #
    # print(all_res)

    rows = []

    for run_num, run_results in all_res.items():
        for timing_model_name, metrics in run_results.items():
            row = {
                "run": run_num,
                "model": timing_model_name,
                **metrics
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("CV_model_results_all_runs_every_participant_logistic.csv", index=False)
    # run_part_c(save_cache=True,
    #            force_recompute_load_data=False,
    #            force_recompute_select_features=True,
    #            force_recompute_find_hp=True,
    #            force_recompute_train_model=True,
    #            force_recompute_test_time_dfs=True,
    #            force_recompute_best_th = True, )
    # all_res = {}
    # for i in range(20):
    #     res, gs = run_part_c(save_cache=True, force_recompute_load_data=False, force_recompute_select_features=True, force_recompute_find_hp=True ,force_recompute_train_model=True, force_recompute_evaluate_model=True)
    #     all_res[tuple(gs)] = res[ModelNames.RANDOM_FOREST]
    #
    # rows = []
    #
    # for run_num, run_results in all_res.items():
    #     for model_name, metrics in run_results.items():
    #         row = {
    #             "run": run_num,
    #             "model": model_name,
    #             **metrics
    #         }
    #         rows.append(row)
    #
    # df = pd.DataFrame(rows)
    # df.to_csv("CV_model_results_all_runs.csv", index=False)

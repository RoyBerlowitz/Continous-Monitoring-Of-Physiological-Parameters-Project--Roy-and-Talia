import numpy as np
import pickle
import time
import os
import copy as copy
from sklearn.model_selection import train_test_split
from Functions import segment_signal, extract_features, split_data, load_cache_or_compute, vet_features_split1, vet_features_split2, load_data, find_best_windows

#from Functions_part_b.filter_feature_selection import select_features
from Functions_part_b.SVM_classifier import perform_PCA, find_best_SVM_parameters, train_SVM
from Functions_part_b.train_model import choose_hyperparameters
from Functions_part_b.Random_forest_model import train_random_forest_classifier
from Functions_part_b.evaluate_model import evaluate_model
from Functions_part_b.train_model import train_model
from Functions_part_b.evaluate_model import  evaluate_model
from Functions_part_b.consts import  ModelNames
#from Functions_part_b.wrapper_feature_selection import wrapper_selection


def run_part_a(data_path, save_cache=False, more_prints=False, force_recompute_load_data=True, force_recompute_seg=True, force_recompute_features=True, force_recompute_splits=True, force_recompute_feature_corr=True, force_recompute_vet_features=True):
    """
    Parameters of run_part_a which can be changed in call below
    ----------
    save_cache : bool
        If True, intermediate results from each section of the assignment are saved
        to disk (cached). These cached results can be reused in future executions.

        If False, no new cache files are saved.

    more_prints : bool
        Controls the verbosity of the function.

        If True, additional status messages are printed during execution.

        If False, output is limited.

    forceRecompute : bool (force_recompute_load_data, force_recompute_seg, force_recompute_features etc
        Controls whether sections of the assignment are recomputed even if cached
        results exist.

        If False:
            - A cached result is used when available, and the corresponding section
              is not recomputed.
            - If no cached result exists, the section is computed normally.

        If True:
            - The section is recomputed regardless of whether a cached result exists.
    """
    data_files = load_cache_or_compute(
        "load_data_output.pkl",
        lambda: load_data(data_path),
        force_recompute=force_recompute_load_data,
        save=save_cache
    )

    print('\033[32mData loaded\033[0m')

    ##--------------- Part A: Segmentation ----------------##
    X_matrix, Y_vector = load_cache_or_compute(
        "segment_output.pkl",
        lambda: segment_signal(7, 0.25*7, data_files, more_prints),
        force_recompute=force_recompute_seg,
        save=save_cache
    )

    print('\033[32mSegmentation completed\033[0m')

    ##--------------- Part B: Feature Extraction -----------##
    X_features = load_cache_or_compute(
        "X_features.pkl",
        lambda: extract_features(X_matrix, data_files, more_prints),
        force_recompute=force_recompute_features,
        save=save_cache
    )

    print('\033[32mFeature extraction completed\033[0m')

    #--------------- Part C: Train & Test ---------------##

    [split1,split2]  = load_cache_or_compute(
        "splits.pkl",
        lambda: split_data(X_features, Y_vector),
        force_recompute=force_recompute_splits,
        save=save_cache
    )

    print('\033[32mTrain test split completed\033[0m')

    #--------------- Part D:  ---------------##
    # This is the feature correlation function to choose best windows
    # This check is long and the process of choosing is detailed in the feature_correlation script - it is not recommended to activate it

    # overlap_options = [0.25]
    # window_duration_options = np.linspace(5, 14, 10)
    # article_features_top_windows_durations, article_features_full_windows_durations_ = load_cache_or_compute(
    #     "article_features_full_windows_durations_.pkl",
    #     lambda: find_best_windows(data_path, window_duration_options, overlap_options, 3, case = "distribution"),
    #     force_recompute=force_recompute_feature_corr,
    #     save=save_cache
    # )
    #
    # frequency_features_top_windows_durations, frequency_features_full_windows_durations_ = load_cache_or_compute(
    #     "frequency_features_full_windows_durations_.pkl",
    #     lambda: find_best_windows(data_path, window_duration_options, overlap_options, 3, case = "frequency"),
    #     force_recompute=force_recompute_feature_corr,
    #     save=save_cache
    # )

    #--------------- Part E: Vetting & Normalization ---------------##

    #split1
    split1_vet_features = load_cache_or_compute(
        "split1_vet_features.pkl",
        lambda: vet_features_split1(split1, more_prints),
        force_recompute=force_recompute_vet_features,
        save=save_cache
    )

    #split2
    split2_vet_features = load_cache_or_compute(
        "split2_vet_features.pkl",
        lambda: vet_features_split2(split2, more_prints),
        force_recompute=force_recompute_vet_features,
        save=save_cache
    )

    print('\033[32mFeature vetting completed\033[0m')

    #save final output
    if save_cache:
        with open("part_a_final_output.pkl", "wb") as f:
            pickle.dump([split1_vet_features, split2_vet_features], f)

    return split1_vet_features, split2_vet_features

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
data_path = os.path.join(script_directory, "data")

if __name__ == "__main__":
    start_time = time.time()
    split1_dfs, split2_dfs = run_part_a(data_path, save_cache=False,more_prints=True, force_recompute_load_data=False, force_recompute_seg=False, force_recompute_features=False, force_recompute_splits=False, force_recompute_vet_features = False)
    #split1_dfs, split2_dfs = run_part_a(data_path, save_cache=True)
    [split2_X_vetting, split2_X_test_norm, split2_Y_train, split2_Y_test, split2_scaler] = split2_dfs
    [split1_X_vetting, split1_X_test_norm, split1_Y_train, split1_Y_test, split1_scaler] = split1_dfs
    # _, split_1_check_df,_,split_1_labels_check_df  = train_test_split(
    #     split1_X_vetting,split1_Y_train,
    #     test_size=1000,
    #     stratify=split1_Y_train,
    #     random_state=42
    # )
    # _, split_2_check_df,_,split_2_labels_check_df  = train_test_split(
    #     split2_X_vetting,split2_Y_train,
    #     test_size=1000,
    #     stratify=split2_Y_train,
    #     random_state=42
    # )

    # split1_X_selected, split1_X_test_norm = select_features(split_1_check_df, split_1_labels_check_df,split1_X_test_norm, split_name="Individual_split", stopping_criteria=0)
    # split2_X_selected, split2_X_test_norm = select_features(split_2_check_df, split_2_labels_check_df,split2_X_test_norm, split_name="Group_split updated", stopping_criteria=0)


    #split1_X_selected, split1_X_test_norm = select_features(split1_X_vetting.head(), split1_Y_train,split1_X_test_norm, split_name="Individual_split", stopping_criteria=0)
    # split2_X_selected, split2_X_test_norm = select_features(split2_X_vetting, split2_Y_train,split2_X_test_norm, split_name="Group_split updated", stopping_criteria=0)



    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    col_split_1 = ['Acc_Z-AXIS_velocity_std', 'Acc_Z-AXIS_kurtosis', 'Mag_MEAN_AXES_CORR', 'Gyro_X-AXIS_CUSUM-_Feature', 'Gyro_X_Z_CORR', 'Acc_Z-AXIS_velocity_median', 'Gyro_Y-AXIS_dominant_frequency', 'Mag_Y-AXIS_skewness', 'Gyro_X-AXIS_CUSUM+_Feature', 'Acc_Y-AXIS_frequency_variance', 'Acc_SM_acceleration_median']
    split1_X_selected = split1_X_vetting[col_split_1 + administrative_features]
    split1_X_test_norm = split1_X_test_norm[col_split_1 + administrative_features]
    # print("adjusted columns")
    # bset_xg_params =  {'colsample_bytree': np.float64(0.6028265220878869), 'gamma': np.float64(0.006918727512424727), 'learning_rate': np.float64(0.05958758776628972), 'max_depth': 9, 'n_estimators': 219, 'scale_pos_weight': np.float64(23.907886017229952), 'subsample': np.float64(0.7168578594140873)}
    # best_xg_boost_for_split1= train_model(split1_X_selected, split1_Y_train, bset_xg_params, ModelNames.XGBOOST)
    # print("finished training XGBoost")
    # xg_score = evaluate_model([best_xg_boost_for_split1], "XG_BOOST_TRY1_SPLIT1", split1_X_test_norm[col_split_1], split1_Y_test, save_model_outputs=True, split_name="just a try individual")
    # # we don't want the administrative features to be a part of the model, so we remove them from the hyperparameteres loop.
    # perform_PCA(split1_X_vetting.drop(administrative_features, axis=1), split2_Y_train, n_dimensions=2, name="Group Split")

    #________________________________________________________________________________________________________
    # לבחור מחדש ולהריץ שוב
    # best_params_for_split_1_RF = {'Random_Forest__class_weight': 'balanced', 'Random_Forest__max_depth': 37, 'Random_Forest__max_samples': np.float64(0.7253152871382157), 'Random_Forest__min_samples_split': 28, 'Random_Forest__n_estimators': 445}
    # wrapper_selection(split1_X_vetting.drop(administrative_features, axis=1), split1_Y_train, best_params_for_split_1_RF,
    #                   n_features_range= [3,5,7,10,12,15,17,19,20],
    #                   model_type='RF', split_name="Individual")
    # best_features_list_split_1 = [
    #     "Acc_Z-AXIS_velocity_std",
    #     "Acc_Z-AXIS_kurtosis",
    #     "Mag_SM_slew_rate",
    #     "Gyro_X_Z_CORR",
    #     "Acc_Z-AXIS_CUSUM+_Feature",
    #     "Gyro_Z-AXIS_dominant_frequency",
    #     "Gyro_SM_velocity_median",
    #     "Acc_Z-AXIS_CUSUM-_Feature",
    #     "Acc_SM_min",
    #     "Acc_SM_acceleration_median",
    #     "Gyro_Y-AXIS_dominant_frequency",
    #     "Acc_Z-AXIS_velocity_median",
    #     "Gyro_Y-AXIS_velocity_median",
    #     "Acc_X_Z_CORR",
    #     "Gyro_X-AXIS_CUSUM-_Feature",
    #     "Acc_Y-AXIS_frequency_variance",
    #     "Gyro_X-AXIS_CUSUM+_Feature"
    # ]

    # best_params_for_split_1_RF = choose_hyperparameters(split1_X_vetting[best_features_list_split_1+administrative_features],
    #                                                     split1_Y_train, model=ModelNames.RANDOM_FOREST, n_jobs=-1,
    #                                                     n_iterations=50, split_name="Individual Split after wrapper")
    # train_rf_split_1 = train_random_forest_classifier(split1_X_vetting[best_features_list_split_1+administrative_features],
    #                                                   split1_Y_train, best_params_for_split_1_RF,
    #                                                   name="RF AFTER WRAPPER - INDIVIDUAL SPLIT")
    #
    # train_rf_split_1_model_output = evaluate_model(train_rf_split_1, ['Random Forest'],
    #                                                split1_X_test_norm.drop(administrative_features, axis=1),
    #                                                split1_Y_test, save_model_outputs=True, split_name="RF AFTER WRAPPER - INDIVIDUAL SPLIT")
    # לבחור מחדש ולהריץ שוב
    # best_params_for_split_2_RF = {'Random_Forest__class_weight': 'balanced', 'Random_Forest__max_depth': 26, 'Random_Forest__max_samples': np.float64(0.78453678109946), 'Random_Forest__min_samples_split': 54, 'Random_Forest__n_estimators': 388}
    # wrapper_selection(split2_X_vetting.drop(administrative_features, axis=1), split2_Y_train, best_params_for_split_2_RF,
    #                   n_features_range= [3,5,7,10,12,15,17,19,20], model_type='RF', split_name="Group 1st wrapper")

    # best_features_list_split_2 = []
    # best_params_for_split_2_RF = choose_hyperparameters(split2_X_vetting[best_features_list_split_2+administrative_features],
    #                                                     split2_Y_train, model=ModelNames.RANDOM_FOREST, n_jobs=-1,
    #                                                     n_iterations=50, split_name="Group Split after wrapper")
    # train_rf_split_2 = train_random_forest_classifier(split2_X_vetting[best_features_list_split_2],
    #                                                   split2_Y_train, best_params_for_split_2_RF,
    #                                                   name="RF AFTER WRAPPER - INDIVIDUAL SPLIT")
    #
    # train_rf_split_2_model_output = evaluate_model([train_rf_split_2], ['Random Forest'],
    #                                                split1_X_test_norm.drop(administrative_features, axis=1),
    #                                                split1_Y_test, save_model_outputs=True, split_name="RF AFTER WRAPPER - Group Split")
    #________________________________________________________________________________________________________

    # best_params_for_split_1_RF = choose_hyperparameters(split1_X_selected,
    #                                                     split1_Y_train, model=ModelNames.RANDOM_FOREST, n_jobs=-1,
    #                                                     n_iterations=50, split_name="Individual Split after wrapper")
    # train_rf_split_1 = train_random_forest_classifier((split1_X_selected.drop(administrative_features, axis=1)),
    #                                                   split1_Y_train, split1_X_selected['Group number'], best_params_for_split_1_RF,
    #                                                   name="Individual Split")
    #
    # train_rf_split_1_model_output = evaluate_model([train_rf_split_1], ['Random Forest'],
    #                                                split1_X_test_norm.drop(administrative_features, axis=1),
    #                                                split1_Y_test, save_model_outputs=True, split_name="Individual - updated check")
    # print("evaluated model output")



    # we don't want the administrative features to be a part of the model, so we remove them from the hyperparameteres loop.
    # perform_PCA(split2_X_vetting.drop(administrative_features, axis=1), split2_Y_train, n_dimensions =2, name="Individual Split")
    # perform_PCA(split1_X_vetting.drop(administrative_features, axis=1), split2_Y_train, n_dimensions=2, name="Group Split")

    split2_X_selected = split2_X_vetting
    # perform_PCA(split2_X_selected, split2_Y_train, n_dimensions =2, name="Group Split updated")

    # best_params_for_split_2_RF = choose_hyperparameters(split2_X_selected,
    #                                                  split2_Y_train, model=ModelNames.RANDOM_FOREST, n_jobs=-1, n_iterations=50,
    #                                                  split_name="Group Split updated", split_by_group_flag = True)

    #best_params_for_split_2_RF = {'Random_Forest__class_weight': 'balanced_subsample', 'Random_Forest__max_depth': 26, 'Random_Forest__max_samples': np.float64(0.6035119926400068), 'Random_Forest__min_samples_leaf': 13, 'Random_Forest__min_samples_split': 21, 'Random_Forest__n_estimators': 489}
    # train_rf_split_2 = train_random_forest_classifier((split2_X_selected.drop(administrative_features, axis=1)),
    #                                                   split2_Y_train, best_params_for_split_2_RF, name="Group Split updated")
    # train_rf_split_2_model_output = evaluate_model([train_rf_split_2], ['Random Forest'], split2_X_test_norm.drop(administrative_features, axis=1), split2_Y_test, save_model_outputs=True, split_name="Group Split updated")
    #
    #
    # # best_params_split1_SVM = find_best_SVM_parameters(split1_X_selected.drop(administrative_features, axis=1), split1_Y_train, n_jobs=8, n_iterations=30, split_name="Individual Split")
    # # train_SVM(split1_X_selected.drop(administrative_features, axis=1), split1_Y_train,  split1_X_test_norm.drop(administrative_features, axis=1), split1_Y_test, best_params_split1_SVM, name="Individual Split")
    # best_params_split2_SVM = find_best_SVM_parameters(split2_X_selected.drop(administrative_features, axis=1), split2_Y_train, split2_X_selected['Group number'], n_jobs=8, n_iterations=30, split_name="Group Split", split_by_group_flag = True)
    # train_SVM(split2_X_selected.drop(administrative_features, axis=1),split2_Y_train,  split2_X_test_norm.drop(administrative_features, axis=1), split2_Y_test, best_params_split2_SVM, name="Group Split")

    best_params_for_split_1_XG = choose_hyperparameters(split1_X_selected,
                                                        split1_Y_train, model=ModelNames.XGBOOST, n_jobs=-1,
                                                        n_iterations=50, split_name="Individual XGBOOST Split after wrapper")
    train_XG_split_1 = train_model(split1_X_selected,split1_Y_train, best_params_for_split_1_XG, ModelNames.XGBOOST, split_by_group_flag = True)


    train_XG_split_1_model_output = evaluate_model([train_XG_split_1], ['XGBOOST individiaul'],
                                                   split1_X_test_norm.drop(administrative_features, axis=1),
                                                   split1_Y_test, save_model_outputs=True, split_name="Individual XGBOOST - updated check")

    best_params_for_split_2_XG = choose_hyperparameters(split2_X_selected,split2_Y_train, model=ModelNames.XGBOOST, n_jobs=-1, n_iterations=50,split_name="Group XGBOOST Split updated", split_by_group_flag = True)
    train_XG_split_2 =  train_model(split2_X_selected,split2_Y_train, best_params_for_split_2_XG, ModelNames.XGBOOST, split_by_group_flag = True)
    train_XG_split_1_model_output = evaluate_model([train_XG_split_2], ['XGBOOST GROUP'],
                                                   split2_X_test_norm.drop(administrative_features, axis=1),
                                                   split2_Y_test, save_model_outputs=True, split_name="Individual XGBOOST - updated check")

    end_time = time.time()
    print(f"Total time: {end_time - start_time} sec")

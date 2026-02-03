from pathlib import Path
import time

from .Functions import *

def run_train(save_cache=False, recompute_functions=RecomputeFunctionsConfig(), group_name = "02"):
    start_time = time.time()

    window_models = [WindowModelNames.XGBOOST, WindowModelNames.RANDOM_FOREST]
    #window_models = [WindowModelNames.XGBOOST] #talia
    window_models = [WindowModelNames.RANDOM_FOREST] #roee

    second_models = [SecondModelNames.NO_MODEL, SecondModelNames.MARKOV] #SecondModelNames.LOGISTIC, decided not to use

    ## ========================================================================================================== ##
    ##                                               PREPROCESSING                                                ##
    ## ========================================================================================================== ##

    ## ==================================== Load Data ==================================== ##
    data_path = Path(__file__).resolve().parent.parent / "data"
    data_files = load_cache(
        "load_data.pkl",
        lambda: load_data(data_path),
        force_recompute=recompute_functions.load_data,
        save=save_cache
    )
    print('\033[32mData loaded\033[0m')

    ## ==================================== Segmentation ==================================== ##
    X_matrix, y_train = load_cache(
        "segment_signal.pkl",
        lambda: segment_signal(7, 0.25 * 7, data_files), #params were chosen in Part A by maximizing MU
        force_recompute=recompute_functions.segment_signal,
        save=save_cache
    )
    print('\033[32mSegmentation completed\033[0m')

    ## ========================================================================================================== ##
    ##                                                FEATURES                                                    ##
    ## ========================================================================================================== ##

    ## ==================================== Feature Extraction ==================================== ##
    X_train = load_cache(
        "extract_features.pkl",
        lambda: extract_features(X_matrix, data_files),
        force_recompute=recompute_functions.extract_features,
        save=save_cache
    )
    print('\033[32mFeature extraction completed\033[0m')

    # ## ==================================== CNN Embedding ==================================== ##
    #informative_features = ['cnn_emb_2', 'cnn_emb_6', 'cnn_emb_5', 'Gyro_Y-AXIS_dominant_frequency', 'Gyro_Y-AXIS_CUSUM-_Feature', 'Acc_X-AXIS_acceleration_std', 'Acc_X_Z_CORR', 'Gyro_SM_velocity_median', 'Gyro_Y-AXIS_CUSUM+_Feature', 'Acc_SM_dominant_frequency', 'Mag_Y-AXIS_median', 'Gyro_Z-AXIS_band_to_tot_energy_ratio', 'Mag_MEAN_AXES_CORR', 'cnn_emb_8', 'Acc_SM_acceleration_median', 'Acc_X-AXIS_velocity_skewness', 'Acc_Z-AXIS_velocity_median', 'Acc_SM_frequency_centroid', 'Acc_Z-AXIS_CUSUM+_Feature', 'Gyro_Y-AXIS_velocity_median']
    informative_features = [ 'Gyro_Y-AXIS_dominant_frequency', 'Gyro_Y-AXIS_CUSUM-_Feature', 'Acc_X-AXIS_acceleration_std', 'Acc_X_Z_CORR', 'Gyro_SM_velocity_median', 'Gyro_Y-AXIS_CUSUM+_Feature', 'Acc_SM_dominant_frequency', 'Mag_Y-AXIS_median', 'Gyro_Z-AXIS_band_to_tot_energy_ratio', 'Mag_MEAN_AXES_CORR', 'Acc_SM_acceleration_median', 'Acc_X-AXIS_velocity_skewness', 'Acc_Z-AXIS_velocity_median', 'Acc_SM_frequency_centroid', 'Acc_Z-AXIS_CUSUM+_Feature', 'Gyro_Y-AXIS_velocity_median']
    X_train, chosen_features = load_cache(
        "cnn_embedding.pkl",
        lambda: cnn_embedding_full_workflow(X_train, y_train, informative_features, group_name),
        force_recompute=recompute_functions.cnn_embedding,
        save=save_cache
    )
    print('\033[32mCNN embedding completed\033[0m')
    print(chosen_features)
    ## ==================================== Normalization ==================================== ##
    [X_train, scaler] = load_cache(
        "feature_normalization.pkl",
        lambda: normalize_train(X_train),
        force_recompute=recompute_functions.feature_normalization,
        save=save_cache
    )
    save_pickle_to_test(scaler, "normalization_train_scaler.pkl")
    print('\033[32mFeature normalization completed\033[0m')

    ## ==================================== Vetting ==================================== ##
    # The reason the vetting is hidden in the comment is that it has already been executed.
    # The features that were selected are embodied in extract_features,
    # which only gives the selected features + the administrative features.

    # chosen_vet_features = load_cache(
    #     "vet_features.pkl",
    #     lambda: vet_features(X_train, y_train),
    #     force_recompute=recompute_functions.vet_features,
    #     save=save_cache
    # )
    # X_train = X_train[chosen_vet_features]
    # print('\033[32mFeature vetting completed\033[0m')

    # #!TODO check normalization

    ## ========================================================================================================== ##
    ##                                               WINDOW MODELS                                                ##
    ## ========================================================================================================== ##

    trained_window_models = {}
    trained_second_models = {}
    X_train_seconds_dfs = {}

    for window_model in window_models:
        trained_second_models[window_model] = {}
        ## ==================================== Wrapper Feature Selection ==================================== ##
        [selected_feats,_] = load_cache(
            f"select_features_{window_model}.pkl",
            lambda: select_features(X_train, y_train, models_hp_for_wrapper[window_model], split_by_group_flag=True),
            force_recompute=recompute_functions.select_features,
            save=save_cache
        )
        save_pickle_to_test(selected_feats, f"select_features_{window_model}_train.pkl")
        print(f'\033[32mFeature selection completed model: {window_model}\033[0m')

        X_selected = X_train[selected_feats + admin_features]

        ## ==================================== Choose Best HP For Window Models ==================================== ##
        model_best_hp = load_cache(
            f"choose_hyperparameters_{window_model}.pkl",
            lambda: choose_hyperparameters(X_selected, y_train, window_model, split_by_group_flag=True),
            force_recompute=recompute_functions.choose_hyperparameters,
            save=save_cache
        )
        print(f'\033[32mChoose hp completed model: {window_model}\033[0m')

        ## ==================================== Train Window Models ==================================== ##
        trained_window_models[window_model], X_train_seconds_dfs[window_model] = load_cache(
            f"train_window_model_{window_model}.pkl",
            lambda: train_window_model(X_selected, y_train, model_best_hp, window_model, split_by_group_flag=True),
            force_recompute=recompute_functions.train_window_model,
            save=save_cache
        )
        save_pickle_to_test(trained_window_models[window_model], f"train_window_model_{window_model}_train.pkl")
        print(f'\033[32mFinished training model: {window_model}\033[0m')

        # # ========================================================================================================== ##
        # #                                                  SECOND MODELS                                             ##
        # # ========================================================================================================== ##
        for second_model in second_models:
            ## ==================================== Train Second Models ==================================== ##
            trained_second_models[window_model][second_model] = load_cache(
                f"train_second_model_{window_model}_{second_model}.pkl",
                lambda: prediction_by_second_train(X_train_seconds_dfs[window_model], data_files, window_model, second_model),
                force_recompute=recompute_functions.train_second_model,
                save=save_cache
            )
            save_pickle_to_test(trained_second_models[window_model][second_model], f"train_second_model_{window_model}_{second_model}_train.pkl")
            print(f'\033[32mFinished training second model: {window_model}-{second_model}\033[0m')

    # ========================================================================================================== ##
    #                                               EVALUATE MODELS                                              ##
    # ========================================================================================================== ##
    for window_model in window_models:
        model_stats = {}
        model_stats['window'] = load_cache(
            f"evaluate_window_model_{window_model}.pkl",
            lambda: evaluate_one_model(trained_window_models[window_model], window_model, X_selected, y_train),
            force_recompute=recompute_functions.evaluate_models,
            save=save_cache
        )
        print(f'\033[32mFinished evaluating model: {window_model}\033[0m')

        for second_model in second_models:
            model_stats[second_model] = load_cache(
                f"evaluate_second_model_{window_model}_{second_model}.pkl",
                lambda: prediction_by_second_test(X_train_seconds_dfs[window_model], data_files, window_model, trained_second_models[window_model][second_model], second_model),
                force_recompute=recompute_functions.evaluate_models,
                save=save_cache
            )
            print(f'\033[32mFinished evaluating second model: {window_model}-{second_model}\033[0m')

        save_second_model_stats(model_stats, window_model)


    end_time = time.time()
    print(f"Total time: {end_time - start_time} sec")

    return chosen_features

# ========================================================= Run =========================================================
# if __name__ == "__main__":
#     grps = ['15A', '27A', '31A', '42A', '58A', '64A', '79A', '93A', '15B', '27B', '31B', '42B', '58B', '64B', '79B', '93B']
#     lst = ['15', '27', '31', '42', '58', '64', '79', '93']
#
#     recompute_functions = RecomputeTrainFunctionsConfig(
#         load_data=False,
#         segment_signal=False,
#         extract_features=False,
#         split_data=False,
#         feature_normalization=False,
#         vet_features=False,
#         select_features=False,
#         choose_hyperparameters=False,
#         train_window_model=False,
#         create_test_time_df=False,
#         prediction_by_second=False,
#     )
#     run_train(save_cache=True, recompute_functions=recompute_functions, train_groups=part_80)


from pathlib import Path
import time

from .Functions import *

def run_predict(save_cache=False, recompute_functions=RecomputeFunctionsConfig()):
    start_time = time.time()

    window_models = [WindowModelNames.XGBOOST, WindowModelNames.RANDOM_FOREST]
    window_models = [WindowModelNames.XGBOOST] #talia
    # window_models = [WindowModelNames.RANDOM_FOREST] #roee

    second_models = [SecondModelNames.NO_MODEL, SecondModelNames.LOGISTIC, SecondModelNames.MARKOV]

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
    X_matrix, y_test = load_cache(
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
    X_features = load_cache(
        "extract_features.pkl",
        lambda: extract_features(X_matrix, data_files),
        force_recompute=recompute_functions.extract_features,
        save=save_cache
    )
    print('\033[32mFeature extraction completed\033[0m')

    ## ==================================== Normalization ==================================== ##
    scaler = load_pickle("normalization_train_scaler.pkl")
    X_test = load_cache(
        "feature_normalization.pkl",
        lambda: normalize_test(X_features, scaler),
        force_recompute=recompute_functions.feature_normalization,
        save=save_cache
    )
    print('\033[32mFeature normalization completed\033[0m')

    # ## ==================================== Vetting ==================================== ##
    # chosen_vet_features = load_pickle("vet_features_train.pkl")
    # X_test = X_test[chosen_vet_features]
    # print('\033[32mFeature vetting completed\033[0m')
    #
        # ## ==================================== Create Test Seconds Data Frame ==================================== ##
        # X_test_seconds_dfs[window_model] = load_cache(
        #     f"test_seconds_df_{window_model}.pkl",
        #     lambda: create_test_time_df(X_test, trained_window_models[window_model], selected_feats),
        #     force_recompute=recompute_functions.create_test_time_df,
        #     save=save_cache
        # )
        #
    # ========================================================================================================== ##
    #                                               EVALUATE MODELS                                              ##
    # ========================================================================================================== ##
    for window_model in window_models:
        selected_feats = load_pickle(f"select_features_{window_model}_train.pkl")
        X_test = X_test[selected_feats+admin_features]
        window_model_trained = load_pickle(f"train_window_model_{window_model}_train.pkl")

        X_test_seconds_dfs = load_cache(
            f"test_seconds_df_{window_model}.pkl",
            lambda: create_test_time_df(X_test, window_model_trained, selected_feats),
            force_recompute=recompute_functions.create_test_time_df,
            save=save_cache
        )

        model_stats = load_cache(
            f"evaluate_window_model_{window_model}.pkl",
            lambda: evaluate_one_model(window_model_trained, window_model, X_test, y_test),
            force_recompute=recompute_functions.evaluate_models,
            save=save_cache
        )
        print(f'\033[32mFinished evaluating model: {window_model}\033[0m')

        for second_model in second_models:
            second_model_trained = load_pickle(f"train_second_model_{window_model}_{second_model}_train.pkl")
            model_stats = load_cache(
                f"evaluate_second_model_{window_model}_{second_model}.pkl",
                lambda: prediction_by_second_test(X_test_seconds_dfs, data_files, window_model, second_model_trained, second_model),
                force_recompute=recompute_functions.evaluate_models,
                save=save_cache
            )
            print(f'\033[32mFinished evaluating second model: {window_model}-{second_model}\033[0m')
            #!TODO add evaluate test
            ## ==== save model outputs?


    end_time = time.time()
    print(f"Total time: {end_time - start_time} sec")

    return

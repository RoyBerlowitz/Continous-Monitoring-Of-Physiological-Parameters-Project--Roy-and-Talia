from pathlib import Path
import time

from Functions_all import *

def run_predict(save_cache=False, recompute_functions=RecomputeFunctionsConfig(), group_name='02'):
    start_time = time.time()

    window_models = [WindowModelNames.XGBOOST, WindowModelNames.RANDOM_FOREST]
    # window_models = [WindowModelNames.XGBOOST] #talia
    window_models = [WindowModelNames.RANDOM_FOREST] #roee

    second_models = [SecondModelNames.NO_MODEL]#, SecondModelNames.MARKOV]

    ## ========================================================================================================== ##
    ##                                               PREPROCESSING                                                ##
    ## ========================================================================================================== ##
    ## ==================================== Load Data ==================================== ##
    data_path = Path(__file__).resolve().parent.parent / "data_test" #!TODO
    data_files = load_cache(
        "load_data.pkl",
        lambda: load_data(data_path),
        force_recompute=recompute_functions.load_data,
        save=save_cache
    )
    print('\033[32mData loaded\033[0m')

    ## ==================================== Segmentation ==================================== ##
    X_matrix = load_cache(
        "segment_signal.pkl",
        lambda: segment_signal_test(7, 0.25 * 7, data_files),  # params were chosen in Part A by maximizing MU
        force_recompute=recompute_functions.segment_signal,
        save=save_cache
    )
    print('\033[32mSegmentation completed\033[0m')

    ## ========================================================================================================== ##
    ##                                                FEATURES                                                    ##
    ## ========================================================================================================== ##

    ## ==================================== Feature Extraction ==================================== ##
    X_test = load_cache(
        "extract_features.pkl",
        lambda: extract_features(X_matrix, data_files),
        force_recompute=recompute_functions.extract_features,
        save=save_cache
    )
    print('\033[32mFeature extraction completed\033[0m')

    # ## ==================================== CNN Embedding ==================================== ##
    X_test = load_cache(
        "cnn_embedding.pkl",
        lambda: cnn_embedding_full_workflow(X_test, [], group_name, test_flag=True),
        force_recompute=recompute_functions.cnn_embedding,
        save=save_cache
    )
    print('\033[32mCNN embedding completed\033[0m')

    ## ==================================== Normalization ==================================== ##
    scaler = load_pickle("normalization_train_scaler.pkl")
    X_test = load_cache(
        "feature_normalization.pkl",
        lambda: normalize_test(X_test, scaler),
        force_recompute=recompute_functions.feature_normalization,
        save=save_cache
    )
    print('\033[32mFeature normalization completed\033[0m')

    # ========================================================================================================== ##
    #                                               EVALUATE MODELS                                              ##
    # ========================================================================================================== ##
    for window_model in window_models:
        model_stats = {}
        selected_feats = load_pickle(f"select_features_{window_model}_train.pkl")
        X_test = X_test[selected_feats+admin_features]
        window_model_trained = load_pickle(f"train_window_model_{window_model}_train.pkl")

        X_test_seconds_dfs = load_cache(
            f"test_seconds_df_{window_model}.pkl",
            lambda: create_test_time_df(X_test, window_model_trained, selected_feats),
            force_recompute=recompute_functions.create_test_time_df,
            save=save_cache
        )

        for second_model in second_models:
            second_model_trained = load_pickle(f"train_second_model_{window_model}_{second_model}_train.pkl")
            model_stats[second_model] = load_cache(
                f"evaluate_second_model_{window_model}_{second_model}.pkl",
                lambda: prediction_test_each_second(X_test_seconds_dfs, data_files, window_model, second_model_trained, second_model),
                force_recompute=recompute_functions.evaluate_models,
                save=save_cache
            )
            print(f'\033[32mFinished evaluating second model: {window_model}-{second_model}\033[0m')

    end_time = time.time()
    print(f"Total time: {end_time - start_time} sec")

    return

# ========================================================= Run =========================================================
if __name__ == "__main__":

    recompute_functions = RecomputeFunctionsConfig(
        # load_data=False,
        # segment_signal=False,
        # extract_features=False,
        # cnn_embedding=False,
        # feature_normalization=False,
        # vet_features=False,
        # select_features=False,
        # choose_hyperparameters=False,
        # train_window_model=False,
        # create_test_time_df=False,
        # train_second_model=False,
        # evaluate_models=False,
    )
    run_predict(save_cache=True, recompute_functions=recompute_functions)
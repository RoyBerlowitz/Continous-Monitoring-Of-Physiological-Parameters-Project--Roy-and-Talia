from pathlib import Path
import pickle
import time

from Functions import *

def run_train(save_cache=False, recompute_functions=RecomputeFunctionsConfig()):
    start_time = time.time()

    window_models = [WindowModelNames.XGBOOST, WindowModelNames.RANDOM_FOREST]
    window_models = [WindowModelNames.XGBOOST] #talia
    # window_models = [WindowModelNames.RANDOM_FOREST] #roee

    second_models = [SecondModelNames.NO_MODEL,SecondModelNames.LOGISTIC, SecondModelNames.MARKOV]

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
    X_matrix, Y_vector = load_cache(
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
    #!TODO: error /Users/talia/PycharmProjects/Continous-Monitoring-Of-Physiological-Parameters-Project--Roy-and-Talia/02_train/Functions/Features/extract_features_helper_functions.py:670: UserWarning: nperseg=256 is greater than signal length max(len(x), len(y)) = 251, using nperseg = 251
             # frequencies, psd = welch(signal, fs=sampling_rate, nperseg=256)
    X_features = load_cache(
        "extract_features.pkl",
        lambda: extract_features(X_matrix, data_files),
        force_recompute=recompute_functions.extract_features,
        save=save_cache
    )
    print('\033[32mFeature extraction completed\033[0m')

    ## ==================================== Split Train & Test ==================================== ##
    all_split_data = load_cache(
        "split_data.pkl",
        lambda: split_data(X_features, Y_vector),
        force_recompute=recompute_functions.split_data,
        save=save_cache
    )
    print('\033[32mTrain test split completed\033[0m')

    ## ==================================== Vetting & Normalization ==================================== ##
    splits_vet_features = load_cache(
        "vet_features.pkl",
        lambda: vet_features_and_normalize(all_split_data),
        force_recompute=recompute_functions.vet_features_and_normalize,
        save=save_cache
    )
    print('\033[32mFeature vetting and normalization completed\033[0m')

    # part_a_res_cache_path = "./pkls/part_a_final_output.pkl"
    # with open(part_a_res_cache_path, "rb") as f:
    #     part_a_res = pickle.load(f)
    #
    # split1_vet_features, split2_vet_features = part_a_res
    # X_train, X_test, y_train, y_test, scaler = split2_vet_features

    X_train, X_test, y_train, y_test, scaler = splits_vet_features

    ## ========================================================================================================== ##
    ##                                               WINDOW MODELS                                                ##
    ## ========================================================================================================== ##

    trained_window_models = {}
    X_train_seconds_dfs = {}
    X_test_seconds_dfs = {}
    model_stats = {}

    for window_model in window_models:
        ## ==================================== Wrapper Feature Selection ==================================== ##
        selected_feats = load_cache(
            f"select_features_{window_model}.pkl",
            lambda: select_features(X_train, y_train, models_hp_for_wrapper[window_model], split_by_group_flag=True),
            force_recompute=recompute_functions.select_features,
            save=save_cache
        )
        print(f'\033[32mFeature selection completed model: {window_model}\033[0m')

        X_selected = X_train[selected_feats + admin_features]
        X_test = X_test[selected_feats + admin_features]

        ## ==================================== Choose Best HP For Window Models ==================================== ##
        model_best_hp = load_cache(
            f"choose_hyperparameters_{window_model}.pkl",
            #!TODO save what we kept for each train groups
            lambda: choose_hyperparameters(X_selected, y_train, window_model, split_by_group_flag=True),
            force_recompute=recompute_functions.choose_hyperparameters,
            save=save_cache
        )
        print(f'\033[32mFeature selection completed model: {window_model}\033[0m')

        ## ==================================== Train Window Models ==================================== ##
        trained_window_models[window_model], X_train_seconds_dfs[window_model] = load_cache(
            f"train_window_model_{window_model}.pkl",
            lambda: train_window_model(X_selected, y_train, model_best_hp, window_model, split_by_group_flag=True),
            force_recompute=recompute_functions.train_window_model,
            save=save_cache
        )
        print(f'\033[32mFinished training model: {window_model}\033[0m')

        ## ==================================== Create Test Seconds Data Frame ==================================== ##
        X_test_seconds_dfs[window_model] = load_cache(
            f"test_seconds_df_{window_model}.pkl",
            lambda: create_test_time_df(X_test, trained_window_models[window_model], selected_feats),
            force_recompute=recompute_functions.create_test_time_df,
            save=save_cache
        )

        ## ========================================================================================================== ##
        ##                                                  SECOND MODELS                                             ##
        ## ========================================================================================================== ##

        for second_model in second_models:
            ## ==================================== Train Second Models ==================================== ##
            model_stats[window_model][second_model] = load_cache(
                f"train_second_model_{window_model}_{second_model}.pkl",
                lambda: prediction_by_second(X_train_seconds_dfs[window_model], X_test_seconds_dfs[window_model], data_files, window_model, second_model),
                force_recompute=recompute_functions.prediction_by_second,
                save=save_cache
            )

            ## ==================================== Evaluate Test ==================================== ##

            #!TODO add evaluate test
            ## ==== save model outputs?


    end_time = time.time()
    print(f"Total time: {end_time - start_time} sec")

    return

# ========================================================= Run =========================================================
if __name__ == "__main__":
    recompute_functions = RecomputeFunctionsConfig(
        load_data=False,
        segment_signal=False,
        extract_features=False,
        split_data=False,
        vet_features_and_normalize=False,
        select_features=False,
        choose_hyperparameters=False,
        train_window_model=False,
        create_test_time_df=False,
        prediction_by_second=False,
    )
    run_train(save_cache=True, recompute_functions=recompute_functions)


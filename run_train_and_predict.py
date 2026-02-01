from pathlib import Path
import random
import pickle
import os

from o02_train.o02_train import run_train
from o02_test.o02_predict import run_predict
from o02_train.Functions import RecomputeFunctionsConfig, load_data, segment_signal, extract_features

def load_cache_2(cache_path, compute_fn, force_recompute=False, save=True):
    os.makedirs('full_run_pkls', exist_ok=True)

    # Full path to cache file
    cache_path = os.path.join('full_run_pkls', cache_path)

    if (not force_recompute) and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Compute the result
    result = compute_fn()

    # Save if requested
    if save:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result

def resave_cache(cache_path, extension, data):
    cache_path = os.path.join(f'./o02_{extension}/pkls', cache_path)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == '__main__':

    #load all data, segment and extract features BEFORE the train test split so they will not need to be recalced each time

    resplit = False
    if resplit:
        ##Load Data
        data_path = Path(__file__).resolve().parent / "data"
        data_files = load_cache_2(
            "load_data.pkl",
            lambda: load_data(data_path),
            force_recompute=False
        )
        print('\033[32mData loaded\033[0m')
        ##Segmentation
        X_matrix, y_vec = load_cache_2(
            "segment_signal.pkl",
            lambda: segment_signal(7, 0.25 * 7, data_files),  # params were chosen in Part A by maximizing MU
            force_recompute=False,
        )
        print('\033[32mSegmentation completed\033[0m')
        ##Feature Extraction
        X_features = load_cache_2(
            "extract_features.pkl",
            lambda: extract_features(X_matrix, data_files),
            force_recompute=False,
        )
        print('\033[32mFeature extraction completed\033[0m')

        #!TODO remove once you run all again
        y_vec[y_vec == 2] = 0

        #Split train & test
        group_numbers = ['15', '27', '31', '42', '58', '64', '79', '93']
        random_state = 42  # any integer for reproducibility
        rng = random.Random(random_state)  # create a local random generator
        shuffled_lst = group_numbers[:]  # copy so original list stays intact
        rng.shuffle(shuffled_lst)

        split_idx = int(0.2 * len(shuffled_lst))
        part_20 = shuffled_lst[:split_idx]  # test
        part_80 = shuffled_lst[split_idx:]  # train
        print(f'Test groups are: {part_20}')
        #
        #save pickles into 02_trian and 02_test folder pkls
        #data_files
        data_files_80 = {k: v for k, v in data_files.items() if k.split('_')[0] in part_80}
        data_files_20 = {k: v for k, v in data_files.items() if k.split('_')[0] in part_20}
        resave_cache("load_data.pkl",'train',data_files_80)
        resave_cache("load_data.pkl",'test',data_files_20)
        # segmentation
        X_train = X_matrix[X_matrix['Group number'].isin(part_80)]
        y_train = y_vec[X_matrix['Group number'].isin(part_80)]
        X_test = X_matrix[X_matrix['Group number'].isin(part_20)]
        y_test = y_vec[X_matrix['Group number'].isin(part_20)]
        # remove protocol from test
        mask = X_test['Protocol'] != 1
        X_test = X_test[mask]
        y_test = y_test[mask]
        resave_cache("segment_signal.pkl", 'train', [X_train, y_train])
        resave_cache("segment_signal.pkl", 'test', [X_test, y_test])
        # X_features
        X_train_feats = X_features[X_features['Group number'].isin(part_80)]
        X_test_feats = X_features[X_features['Group number'].isin(part_20)]
        #remove protocol from test
        mask = X_test_feats['Protocol'] != 1
        X_test_feats = X_test_feats[mask]
        resave_cache("extract_features.pkl", 'train', X_train_feats)
        resave_cache("extract_features.pkl", 'test', X_test_feats)

    print(f'\033[34mStarting on train ==========================================\033[0m')
    recompute_functions = RecomputeFunctionsConfig(
                load_data=False,
                segment_signal=False,
                extract_features=False,
                split_data=False,
                feature_normalization=False,
                vet_features=False,
                select_features=False,
                choose_hyperparameters=False,
                train_window_model=False,
                create_test_time_df=False,
                train_second_model=False,
                evaluate_models=False,
            )
    run_train(save_cache=True, recompute_functions=recompute_functions)

    print(f'\033[34mStarting on test ==========================================\033[0m')
    recompute_functions = RecomputeFunctionsConfig(
        load_data=False,
        segment_signal=False,
        extract_features=False,
        split_data=False,
        feature_normalization=False,
        vet_features=False,
        select_features=False,
        choose_hyperparameters=False,
        train_window_model=False,
        create_test_time_df=False,
        train_second_model=False,
        evaluate_models=False,
    )
    run_predict(save_cache=True, recompute_functions=recompute_functions)
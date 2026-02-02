from pathlib import Path
import shutil
import pickle
import os
import re

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

def copy_all_model_outputs(group_name):
    base_dir = Path(__file__).resolve().parent / "full_run_model_outputs"
    pattern = re.compile(rf"({re.escape(group_name)})_(\d+)$")

    max_num = 0

    for name in os.listdir(base_dir):
        match = pattern.match(name)
        if match:
            num = int(match.group(2))
            max_num = max(max_num, num)

    new_num = max_num + 1
    new_folder = f"{group_name}_{new_num}"
    new_path = os.path.join(base_dir, new_folder)

    os.makedirs(new_path, exist_ok=True)

    print(f"Created folder: {new_folder}")

    folders_to_copy = ['pkls','run_outputs']

    for folder in folders_to_copy:
        #train
        src = Path(__file__).resolve().parent / 'o02_train' / folder
        dst = Path(__file__).resolve().parent / "full_run_model_outputs" / new_folder / f"train_{folder}"
        shutil.copytree(src, dst)
        #test
        src = Path(__file__).resolve().parent / 'o02_test' / folder
        dst = Path(__file__).resolve().parent / "full_run_model_outputs" / new_folder / f"test_{folder}"
        shutil.copytree(src, dst)

    print(f'Finished folder copy for run {new_folder}')
    return

if __name__ == '__main__':

    #load all data, segment and extract features BEFORE the train test split so they will not need to be recalced each time
    grps = ['15A', '27A', '31A', '42A', '58A', '64A', '79A', '93A', '15B', '27B', '31B', '42B', '58B', '64B', '79B', '93B']
    test_grp = '42B'# grps[0]
    test_grp_num = test_grp[0:2]
    test_grp_id = test_grp[2]
    print(f'Test groups is: {test_grp}')

    resplit = True
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
        #Feature Extraction
        X_features = load_cache_2(
            "extract_features.pkl",
            lambda: extract_features(X_matrix, data_files),
            force_recompute=False,
        )
        print('\033[32mFeature extraction completed\033[0m')
        #
        # #
        # #Split train & test
        # #save pickles into 02_trian and 02_test folder pkls
        # #data_files
        # #!TODO remive teh 42
        # data_files_80 = {k: v for k, v in data_files.items() if (k!='42B' and k!='42A')and(k.split('_')[0] != test_grp_num or (k.split('_')[0] == test_grp_num and k.split('_')[2] != test_grp_id))}
        # data_files_20 = {k: v for k, v in data_files.items() if (k.split('_')[0] == test_grp_num and k.split('_')[2] == test_grp_id)}
        # resave_cache("load_data.pkl",'train',data_files_80)
        # resave_cache("load_data.pkl",'test',data_files_20)
        # # segmentation
        # test_mask = (X_matrix['Group number'] == test_grp_num) & (X_matrix['Participant ID'] == test_grp_id)
        # X_train = X_matrix.loc[~test_mask].copy()
        # y_train = y_vec.loc[~test_mask].copy()
        # #!TODO remove
        # mask42 = X_train['Group number'] != 42
        # X_train = X_train[mask42]
        # y_train = y_train[mask42]
        # X_test = X_matrix.loc[test_mask].copy()
        # y_test = y_vec.loc[test_mask].copy()
        # # remove protocol from test
        # mask = X_test['Protocol'] != 1
        # X_test = X_test[mask]
        # y_test = y_test[mask]
        # resave_cache("segment_signal.pkl", 'train', [X_train, y_train])
        # resave_cache("segment_signal.pkl", 'test', [X_test, y_test])
        # # # X_features
        # # test_mask = (X_features['Group number'] == test_grp_num) & (X_features['Participant ID'] == test_grp_id)
        # # X_train_feats = X_features.loc[~test_mask].copy()
        # # X_test_feats = X_features.loc[test_mask].copy()
        # # #remove protocol from test
        # # mask = X_test_feats['Protocol'] != 1
        # # X_test_feats = X_test_feats[mask]
        # # resave_cache("extract_features.pkl", 'train', X_train_feats)
        # # resave_cache("extract_features.pkl", 'test', X_test_feats)

    # print(f'\033[34mStarting on train ==========================================\033[0m')
    # recompute_functions = RecomputeFunctionsConfig(
    #             load_data=False,
    #             segment_signal=False,
    #             extract_features=False,
    #             feature_normalization=False,
    #             cnn_embedding=False,
    #             vet_features=False,
    #             select_features=False,
    #             choose_hyperparameters=False,
    #             train_window_model=False,
    #             create_test_time_df=False,
    #             train_second_model=False,
    #             evaluate_models=False,
    #         )
    # run_train(save_cache=True, recompute_functions=recompute_functions, group_name=test_grp)
    #
    # print(f'\033[34mStarting on test ==========================================\033[0m')
    # recompute_functions = RecomputeFunctionsConfig(
    #     load_data=False,
    #     segment_signal=False,
    #     extract_features=False,
    #     # cnn_embedding=False,
    #     # feature_normalization=False,
    #     # vet_features=False,
    #     # select_features=False,
    #     # choose_hyperparameters=False,
    #     # train_window_model=False,
    #     # create_test_time_df=False,
    #     # train_second_model=False,
    #     # evaluate_models=False,
    # )
    # run_predict(save_cache=True, recompute_functions=recompute_functions, group_name=test_grp)
    #
    # copy_all_model_outputs(test_grp)
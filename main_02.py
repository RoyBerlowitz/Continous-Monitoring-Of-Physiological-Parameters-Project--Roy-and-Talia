import numpy as np
import time
import os

from Functions import segment_signal, extract_features, split_data, load_cache_or_compute, vet_features, vet_features_split1, load_data, find_best_windows

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
    X_train, X_test, y_train, y_test = split2
    split2_vet_features = load_cache_or_compute(
        "split2_vet_features.pkl",
        lambda: vet_features(X_train, X_test, y_train, more_prints, split_name="Group normalization"),
        force_recompute=force_recompute_vet_features,
        save=save_cache
    )
    split2_vet_features = list(split2_vet_features)
    split2_vet_features.append(y_train)
    split2_vet_features.append(y_test)

    print('\033[32mFeature vetting completed\033[0m')

    return split1_vet_features, split2_vet_features

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
data_path = os.path.join(script_directory, "data")

if __name__ == "__main__":
    start_time = time.time()
    # split1_dfs, split2_dfs = run_part_a(data_path, save_cache=True,more_prints=True, force_recompute_load_data=False, force_recompute_seg=False, force_recompute_features=False, force_recompute_splits=False, force_recompute_vet_features = False)
    split1_dfs, split2_dfs = run_part_a(data_path, save_cache=True)

    end_time = time.time()
    print(f"Total time: {end_time - start_time} sec")

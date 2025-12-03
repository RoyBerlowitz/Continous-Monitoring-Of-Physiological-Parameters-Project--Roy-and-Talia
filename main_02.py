import pandas as pd
import os

from Functions import segment_signal, extract_features, split_data, load_cache_or_compute, vet_features, vet_features_split1, load_data

#cosnts
#todo change before handing in
is_dev = True #False
def run_part_a(data_path, force_recompute_seg=True, force_recompute_features=True, force_recompute_splits=True, force_recompute_vet_features=True):
    data_files = load_data(data_path)

    ##--------------- Part A: Segmentation ----------------##
    X_matrix, Y_vector = load_cache_or_compute(
        "segment_output.pkl",
        lambda: segment_signal(data_path, 50, 25, data_files),
        force_recompute=force_recompute_seg,
        save=is_dev
    )

    ##--------------- Part B: Feature Extraction -----------##
    X_features = load_cache_or_compute(
        "X_features.pkl",
        lambda: extract_features(data_path, X_matrix, data_files),
        force_recompute=force_recompute_features,
        save=is_dev
    )

    print("completed")

    #--------------- Part C: Train & Test ---------------##
    splits = load_cache_or_compute(
        "splits.pkl",
        lambda: split_data(X_features, Y_vector),
        force_recompute=force_recompute_splits,
        save=is_dev
    )

    # --------------- Part E: Vetting & Normalization ---------------##
    split1, split2 = splits
    #split1
    split1_vet_features = load_cache_or_compute(
        "split1_vet_features.pkl",
        lambda: vet_features_split1(split1),
        force_recompute=force_recompute_vet_features,
        save=is_dev
    )

    #split2
    X_train, X_test, y_train, y_test = split2
    split2_vet_features = load_cache_or_compute(
        "split2_vet_features.pkl",
        lambda: vet_features(X_train, X_test, y_train),
        force_recompute=force_recompute_vet_features,
        save=is_dev
    )
    split2_vet_features.append(y_train)
    split2_vet_features.append(y_test)


    return X_features, Y_vector

data_path = r"C:\Users\nirei\OneDrive\Desktop\Bachelors Degree - Biomedical Engineering And Neuroscience\Year 4\Semester A\Continuous Monitoring of Physiological Parameters\PythonProject7\02"
# data_path = r"/Users/talia/Downloads/02 copy 2"
# X_features, Y_vector = run_part_a(data_path)
X_features, Y_vector = run_part_a(data_path, force_recompute_seg=False, force_recompute_features=False, force_recompute_splits=True)
# X_features.to_excel("test.xlsx",index=False)

#final_x = vet_features(X_features, Y_vector, split_name = "Individual Normalization", N=3, K= 10, threshold=0.8)
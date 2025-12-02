import pandas as pd
import os

from Functions import segment_signal, extract_features, split_data, load_cache_or_compute, vet_features, load_data
from Functions.vet_features import vet_features

#cosnts
#todo change before handing in
is_dev = True #False
def run_part_a(data_path, force_recompute_seg=False, force_recompute_features=True, force_recompute_splits=True):
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

    #-------Part C: Train & Test -------
    splits = load_cache_or_compute(
        "splits.pkl",
        lambda: split_data(X_features, Y_vector),
        force_recompute=force_recompute_splits,
        save=is_dev
    )

    return X_features, Y_vector
# אלו השוורת שצריכות להופיע בסוף כדי שיהיה לנו את הpath הנכון
# script_path = os.path.abspath(__file__)
# script_directory = os.path.dirname(script_path)
# data_path = script_directory + "/data"

data_path = r"C:\Users\nirei\OneDrive\Desktop\Bachelors Degree - Biomedical Engineering And Neuroscience\Year 4\Semester A\Continuous Monitoring of Physiological Parameters\PythonProject7\02"
# data_path = r"/Users/talia/Downloads/02"
X_features, Y_vector = run_part_a(data_path, force_recompute_seg=True, force_recompute_features=True, force_recompute_splits=True)
X_features.to_excel("test.xlsx",index=False)

#final_x = vet_features(X_features, Y_vector, split_name = "Individual Normalization", N=3, K= 10, threshold=0.8)
import pandas as pd
import os
import re

from .window_timing_translator_preprocessing import apply_smoothing
from .timing_classifying_without_model import print_metrics_table

def evaluate_test_by_second(X_test, y_test, threshold_no_median, threshold_with_median, filter_size):
    y_probs = X_test["weighted_prob"]
    pred_y_no_median_filter =  (y_probs >= threshold_no_median).astype(int)
    pred_y_with_median_filter =  (y_probs >= threshold_with_median).astype(int)
    smoothing_temp_df = pd.DataFrame({
        'recording_identifier': X_test['recording_identifier'].values,
        'prediction': pred_y_with_median_filter
    })
    smoothed_prediction = apply_smoothing(smoothing_temp_df, filter_size)['smoothed_prediction']

    no_smoothing = print_metrics_table(y_test, pred_y_no_median_filter,"Metrics Table For Chosen Threshold Before Median Filtering TEST")
    with_smoothing = print_metrics_table(y_test, smoothed_prediction, "Metrics Table For Chosen Threshold After Median Filtering TEST")

    return {'test_no_smoothing': no_smoothing, 'test_with_smoothing': with_smoothing}

def save_all_stats(all_stats, model_name):
    folder_name = create_folder_for_saving(model_name)

    df = pd.DataFrame.from_dict(all_stats, orient="index")
    df.index.name = "res_type"

    df.to_excel(f"{folder_name}/model_results.xlsx")

def create_folder_for_saving(split_name):
    base_dir = "model_outputs"

    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Find existing folders that start with a number_
    existing = os.listdir(base_dir)
    indices = []

    for folder in existing:
        match = re.match(r"(\d+)_", folder)
        if match:
            indices.append(int(match.group(1)))

    # Determine next index
    next_index = max(indices) + 1 if indices else 1

    # Create new folder
    folder_path = os.path.join(base_dir, f"{next_index}_{split_name}")
    os.makedirs(folder_path)

    return folder_path

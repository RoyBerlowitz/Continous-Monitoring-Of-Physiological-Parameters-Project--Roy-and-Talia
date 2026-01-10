import pandas as pd
import os
import re

from .window_timing_translator_preprocessing import apply_smoothing
from .timing_classifying_without_model import print_metrics_table

def evaluate_test_by_second_no_model(X_test, y_test, threshold_no_median, threshold_with_median, filter_size):
    # This function is meant to get the results for the model
    y_probs = X_test["weighted_prob"]
    # we compute the labels with the threshold found for the without-filtering scheme
    pred_y_no_median_filter =  (y_probs >= threshold_no_median).astype(int)
    # we compute the labels with the threshold found for the with-filtering scheme
    pred_y_with_median_filter =  (y_probs >= threshold_with_median).astype(int)
    smoothing_temp_df = pd.DataFrame({
        'second': X_test['second'].values,
        'recording_identifier': X_test['recording_identifier'].values,
        'prediction': pred_y_with_median_filter,
        'no_median_prediction': pred_y_no_median_filter,
    })
    # we preform the smoothing
    smoothed_prediction = apply_smoothing(smoothing_temp_df, filter_size)['smoothed_prediction']
    # adding the label
    smoothing_temp_df['true_label'] =  y_test.values
    # removing the prediction without the median filtering
    smoothing_temp_df = smoothing_temp_df.drop(columns=['prediction'])

    # to see the per-second classification, we iterate over each recording and create a df to compare the true label to the prediction with and without median filtering
    recording_dict = {}
    # we get the unique recording identifiers
    for recording in smoothing_temp_df['recording_identifier'].unique():
        # we extract the per-second results
        recording_dict[recording] = smoothing_temp_df[smoothing_temp_df['recording_identifier'] == recording].copy()

    # we get the metrics for with and without smoothing
    no_smoothing = print_metrics_table(y_test, pred_y_no_median_filter,"Metrics Table For Chosen Threshold Before Median Filtering TEST")
    with_smoothing = print_metrics_table(y_test, smoothed_prediction, "Metrics Table For Chosen Threshold After Median Filtering TEST")

    return {'test_no_smoothing': no_smoothing, 'test_with_smoothing': with_smoothing}, recording_dict

def evaluate_test_by_second_with_model(X_test, y_test, model, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    chosen_threshold = model.optimal_threshold_PRC_
    predicted_y = (y_probs >= chosen_threshold).astype(int)
    result_df = pd.DataFrame({
        'recording_identifier': X_test['recording_identifier'].values,
        'second': X_test['second'].values,
        'prediction': predicted_y,
        'true_label': y_test.values,
    })
    # to see the per-second classification, we iterate over each recording and create a df to compare the true label to the prediction with and without median filtering
    recording_dict = {}
    # we get the unique recording identifiers
    for recording in smoothing_temp_df['recording_identifier'].unique():
        # we extract the per-second results
        recording_dict[recording] = result_df[result_df['recording_identifier'] == recording].copy()

    results =   print_metrics_table(y_test, predicted_y,f"Metrics Table For Chosen Threshold for {model_name}  - TEST")
    return results, recording_dict


def save_all_stats(all_stats, model_name, recording_dict):
    # this function meant to save the results
    # creation of the folder and the path
    folder_name = create_folder_for_saving(model_name)
    file_path = f"{folder_name}/model_results.xlsx"

    # the main df with the statistics
    df_main = pd.DataFrame.from_dict(all_stats, orient="index")
    df_main.index.name = "res_type"

    # we want to create several sheets, one with the metrics and others with the classification per second
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # the main sheet is the metric sheet
        df_main.to_excel(writer, sheet_name="Overall_Summary")

        # we iterate over recording dist
        for rec_name, rec_data in recording_dict.items():
            # we transform the dict per recording to df
            recording_df = pd.DataFrame(rec_data)

            # we define the name to be the recording identifier
            clean_sheet_name = rec_name
            # we add the sheets
            recording_df.to_excel(writer, sheet_name=clean_sheet_name, index=False)

    print(f"--- All results and recordings saved to: {file_path} ---")
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

from pathlib import Path
import pandas as pd
import numpy as np
import os
import re

from .window_timing_translator_preprocessing import apply_smoothing
from .markov_model import prepare_data_for_hmm
from .timing_classifying_without_model import print_metrics_table
from ..consts import SecondModelNames

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
    no_smoothing = print_metrics_table(y_test, pred_y_no_median_filter,"Metrics Table For Chosen Threshold Before Median Filtering - Validation")
    with_smoothing = print_metrics_table(y_test, smoothed_prediction, "Metrics Table For Chosen Threshold After Median Filtering - Validation")

    # chose smoothing
    # return {'test_no_smoothing': no_smoothing, 'test_with_smoothing': with_smoothing}, recording_dict
    return {'test_with_smoothing': with_smoothing}, recording_dict



def apply_median_on_test(X_test, threshold_no_median, threshold_with_median, filter_size):
    # This function is meant to get the results for the model - When no real labels available
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
    res_df = apply_smoothing(smoothing_temp_df, filter_size)
    csv_save_path = Path(__file__).resolve().parent.parent.parent

    for recording_id, group in res_df.groupby("recording_identifier"):
        out_df = group.copy()

        out_df["Start"] = out_df["second"]
        out_df["End"] = out_df["second"] + 1
        out_df["Label"] = out_df["smoothed_prediction"]

        out_df = out_df[["Start", "End", "Label"]]

        out_df.to_csv(
            f"{csv_save_path}/{recording_id}_pred.csv",
            index=False
        )
    return res_df

def evaluate_test_by_second_with_model(X_test, y_test, model, model_name, classification_flag = SecondModelNames.LOGISTIC):
    # we get the results for the model
    test_for_calculation = X_test[["prob_1", "prob_2", "prob_3", "prob_4"]]
    if classification_flag == SecondModelNames.LOGISTIC:
        # we extract the probabilities from the model
        y_prob = model.predict_proba(test_for_calculation)[:, 1]
        chosen_threshold = model.optimal_threshold_PRC_
        # based on the found optimal threshold, we classify each time point
        predicted_y = (y_prob >= chosen_threshold).astype(int)
    elif classification_flag == SecondModelNames.MARKOV:
        # we adjust X_test to be sorted correctly, as it is important in markov
        X_test = X_test.sort_values(['recording_identifier', 'second'])
        # we pre-process the data for markov
        X_probs, y_true, lengths_test = prepare_data_for_hmm(X_test, y_test)
        # we compute the log probs exactly as in the train
        _, posteriors = model.score_samples(X_probs, lengths_test)
        epsilon = 1e-15
        log_posteriors = np.log(posteriors + epsilon)
        # we get the transition log probs
        llr_test = log_posteriors[:, 1] - log_posteriors[:, 0]
        # we debug to see the max and min LLR and the chosen threshold
        print(f"DEBUG MARKOV TEST: Max LLR={np.max(llr_test):.2f}, Threshold={model.optimal_threshold_PRC_:.2f}")
        # we obtain the predictions
        predicted_y = (llr_test >= model.optimal_threshold_PRC_).astype(int)

    # we create the results data frame
    result_df = pd.DataFrame({
        'recording_identifier': X_test['recording_identifier'].values,
        'second': X_test['second'].values,
        'prediction': predicted_y,
        'true_label': y_true,
    })
    # to see the per-second classification, we iterate over each recording and create a df to compare the true label to the prediction with and without median filtering
    recording_dict = {}
    # we get the unique recording identifiers
    for recording in result_df['recording_identifier'].unique():
        # we extract the per-second results
        recording_dict[recording] = result_df[result_df['recording_identifier'] == recording].copy()
    # these are the results of classification
    results = print_metrics_table(y_test, predicted_y, f"Metrics Table For Chosen Threshold for {model_name}  - Validation")
    return results, recording_dict

def save_all_stats(all_stats, model_name, recording_dict):
    # this function meant to save the results
    # creation of the folder and the path
    # folder_name = create_folder_for_saving(model_name)
    file_path = Path(__file__).resolve().parent.parent.parent / 'run_outputs' / f"{model_name}_model_results.xlsx"

    # the main df with the statistics
    df_main = pd.DataFrame.from_dict(all_stats, orient="index")
    df_main.index.name = "res_type"

    # we want to create several sheets, one with the metrics and others with the classification per second
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # the main sheet is the metric sheet
        df_main.to_excel(writer, sheet_name="Overall_Summary")
        all_recordings_list = []

        # we iterate over recording dist
        for rec_name, rec_data in recording_dict.items():
            if "second" in rec_data:
                rec_data["Start"] = rec_data.pop("second")

            recording_df = pd.DataFrame(rec_data)

            recording_df["End"] = recording_df["Start"] + 1

            cols = ["Start", "End"] + [c for c in recording_df.columns if c not in ["Start", "End"]]
            recording_df = recording_df[cols]

            # we define the name to be the recording identifier
            clean_sheet_name = rec_name
            # we add the sheets
            # recording_df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
            all_recordings_list.append(recording_df)

        final_df = pd.concat(all_recordings_list, ignore_index=True)



    cols_to_keep = [c for c in ["Start", "End", "true_label", "recording_identifier"] if c in final_df.columns]
    real_df_final = final_df[cols_to_keep].copy()
    real_df_final.rename(columns={'true_label': 'label'}, inplace=True)


    pred_df_final = final_df.drop(columns=["true_label", "no_median_prediction"], errors='ignore')
    real_df_final.rename(columns={'prediction': 'label'}, inplace=True)
    save_path = Path(__file__).resolve().parent.parent.parent / 'run_outputs'
    pred_df_final.to_csv(f"{save_path}/02_train_pred.csv", index=False)
    real_df_final.to_csv(f"{save_path}/02_train_label.csv", index=False)

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

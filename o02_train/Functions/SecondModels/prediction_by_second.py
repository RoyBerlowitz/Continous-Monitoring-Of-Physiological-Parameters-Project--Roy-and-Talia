import pandas as pd
import numpy as np
from .window_timing_translator_preprocessing import get_handwashing_times, apply_smoothing
from .timing_classifying_without_model import translate_prediction_into_time_point_prediction_with_weights, train_for_decision, print_metrics_table
from .evaluate_test_by_second import evaluate_test_by_second_no_model,evaluate_test_by_second_with_model, save_all_stats
from .timing_classifying_with_model import translate_prediction_into_time_point_prediction_for_model, logistic_regression_for_second_classification, train_markov_model
from ..consts import SecondModelNames

def prediction_by_second(train_df, test_df, data_files, model_name, classification_flag = SecondModelNames.NO_MODEL, weight_flag = "Gaussian Weight"):
    # This the function that preforms the final classification to seconds
    # we obtain the handwashing seconds for each train and test df
    train_df = get_handwashing_times(train_df, data_files)
    test_df = get_handwashing_times(test_df, data_files)
    # if we choose to classify with no model
    if classification_flag == SecondModelNames.NO_MODEL:
        train_x, train_y = translate_prediction_into_time_point_prediction_with_weights (train_df, weight_flag)
        test_x, test_y = translate_prediction_into_time_point_prediction_with_weights (test_df, weight_flag)
        # we preform the choice of the threshold
        group_indicator = train_x['Group number'].astype(str) + "_" + train_x['Participant ID'].astype(str)
        threshold_no_median, threshold_with_median, filter_size, train_stats = train_for_decision(train_x, train_y, group_indicator=group_indicator , n_iteration=50, n_jobs=-1)

        #train predictions
        # y_probs = train_x["weighted_prob"]
        # pred_y_no_median_filter =  (y_probs >= threshold_no_median).astype(int)
        # pred_y_with_median_filter =  (y_probs >= threshold_with_median).astype(int)
        # smoothing_temp_df = pd.DataFrame({
        #     'recording_identifier': train_x['recording_identifier'].values,
        #     'prediction': pred_y_with_median_filter
        # })
        # smoothed_prediction = apply_smoothing(smoothing_temp_df, filter_size)['smoothed_prediction']

        #test predictions and eval
        test_stats, recording_dict = evaluate_test_by_second_no_model(test_x, test_y, threshold_no_median, threshold_with_median, filter_size)
        # we save evaluation metrics
        all_stats = {**train_stats, **test_stats}
        save_all_stats(all_stats, model_name+'_no_model_for_second_classification', recording_dict)
        return all_stats
    else:
        train_x, train_y = translate_prediction_into_time_point_prediction_for_model (train_df, weight_flag=None)
        test_x, test_y = translate_prediction_into_time_point_prediction_for_model (test_df, weight_flag=None)
        if classification_flag == SecondModelNames.LOGISTIC:
            model = logistic_regression_for_second_classification(train_x, train_y)
            test_stats, recording_dict = evaluate_test_by_second_with_model(test_x, test_y, model, model_name+'_LR_second_classification')
            all_stats = { **test_stats}
            save_all_stats(all_stats, model_name+'_LR_second_classification', recording_dict)
            return all_stats
        if classification_flag == SecondModelNames.MARKOV:
            model = train_markov_model(train_x, train_y)
            test_stats, recording_dict = evaluate_test_by_second_with_model(test_x, test_y, model, model_name + '_markov_second_classification')
            all_stats = {**test_stats}
            save_all_stats(all_stats, model_name + '_markov_second_classification', recording_dict)
            return all_stats

def prediction_by_second_test(test_df, data_files, model_name, classification_model, classification_flag = SecondModelNames.NO_MODEL, weight_flag = "Gaussian Weight"):
    test_df = get_handwashing_times(test_df, data_files)
    if classification_flag == SecondModelNames.NO_MODEL:
        test_x, test_y = translate_prediction_into_time_point_prediction_with_weights(test_df, weight_flag)
        test_stats, recording_dict = evaluate_test_by_second_no_model(test_x, test_y, classification_model["threshold_no_median"], classification_model["threshold_with_median"], classification_model["filter_size"])
        save_all_stats(test_stats, model_name + '_no_model_for_second_classification', recording_dict)
        return test_stats
    else:
        test_x, test_y = translate_prediction_into_time_point_prediction_for_model(test_df, weight_flag=None)
        if classification_flag == SecondModelNames.LOGISTIC:
            test_stats, recording_dict = evaluate_test_by_second_with_model(test_x, test_y, classification_model, model_name+'_LR_second_classification', classification_flag=classification_flag)
            # all_stats = { **test_stats}
            save_all_stats(test_stats, model_name+'_LR_second_classification', recording_dict)
            return test_stats
        if classification_flag == SecondModelNames.MARKOV:
            test_stats, recording_dict = evaluate_test_by_second_with_model(test_x, test_y, classification_model, model_name + '_markov_second_classification', classification_flag=classification_flag)
            # all_stats = {**test_stats}
            save_all_stats(test_stats, model_name + '_markov_second_classification', recording_dict)
            return test_stats
    return


def prediction_by_second_train(train_df, data_files, model_name, classification_flag = SecondModelNames.NO_MODEL, weight_flag = "Gaussian Weight"):
    # This the function that preforms the final classification to seconds
    # we obtain the handwashing seconds for each train and test df
    train_df = get_handwashing_times(train_df, data_files)
    # if we choose to classify with no model
    if classification_flag == SecondModelNames.NO_MODEL:
        train_x, train_y = translate_prediction_into_time_point_prediction_with_weights (train_df, weight_flag)
        # we preform the choice of the threshold
        parts = train_x['recording_identifier'].str.split('_', expand=True)
        group_indicator = parts[0] + '_' + parts[2]
        threshold_no_median, threshold_with_median, filter_size, train_stats = train_for_decision(train_x, train_y, group_indicator=group_indicator , n_iteration=50, n_jobs=-1)
        # we save evaluation metrics
        # all_stats = {**train_stats}
        # save_all_stats(all_stats, model_name+'_no_model_for_second_classification', recording_dict)
        return {"threshold_no_median":threshold_no_median, "threshold_with_median":threshold_with_median, "filter_size":filter_size, "train_stats":train_stats}
    else:
        train_x, train_y = translate_prediction_into_time_point_prediction_for_model (train_df, weight_flag=None)
        if classification_flag == SecondModelNames.LOGISTIC:
            return logistic_regression_for_second_classification(train_x, train_y)
        if classification_flag == SecondModelNames.MARKOV:
            return train_markov_model(train_x, train_y)

    return
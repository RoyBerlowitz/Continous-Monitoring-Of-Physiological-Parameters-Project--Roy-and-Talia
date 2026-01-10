import pandas as pd
import numpy as np
import math
from Functions_part_c.window_timing_translator_preprocessing import calculate_window_times
from Functions_part_c.timing_classifying_without_model import calculate_time_point_weights
from Functions_part_b.logistic_regression_model import train_logistic_regression, find_best_hp_logistic_regression

from .choose_thresholds import get_threshold_median, get_absolute_threshold_raw, print_metrics_table
from sklearn.metrics import f1_score, precision_recall_curve
from scipy.ndimage import median_filter
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed





def translate_prediction_into_time_point_prediction_for_model (windows_df, weight_flag = None ):
    # here, we take the actual data and translate it to seconds, with the matching weights
    # we start by obtaining all the unique recording identifier, as this will be done on each ientifier seperately
    recordings =  windows_df['recording_identifier'].unique()
    # we will create a new df which is based on seconds instead of windows
    seconds_df = []
    # we iterate over all the recording
    for recording in recordings:
        # we obtain each recording data
        recording_data = windows_df[windows_df['recording_identifier'] == recording]
        # we go over all the complete seconds from the start to end of each recording
        # recording_seconds = range(int(math.floor(recording_data['window_starting_point'].min())), int(math.floor(recording_data['window_ending_point'].max() + 1)))
        recording_seconds = range(int(math.floor(recording_data['First second of the activity'].min())), int(math.floor(recording_data['Last second of the activity'].max() + 1)))
        # creates a dict whose keys are the seconds and the assigned values are None
        dict_of_sec_vals = dict.fromkeys(recording_seconds)
        # we get the exact seconds in which handwashing was operated
        handwashing_times = recording_data["Handwashing time"].iloc[0]

        # we iterate over every row of the recording data, which in that context is a window
        for index, row in recording_data.iterrows():
            # we find the weights for each second in the window
            # window_time_list = calculate_time_point_weights (row["window_times"], row['window_starting_point'], row['window_ending_point'], weights_method = weight_flag)
            window_time_list = calculate_time_point_weights(row["window_times"], row['First second of the activity'], row['Last second of the activity'], weights_method = weight_flag)
            # we get the probability which was already predicted by model
            window_prob = row["window_probability"]
            for sec_dict in window_time_list:
                second = sec_dict["time_point"]
                weight = sec_dict["weight"]
                coverage = sec_dict["coverage"]
                # if this is the first time we see this second - we create the dict
                if dict_of_sec_vals[second] is None:
                    dict_of_sec_vals[second] = {"prob": [window_prob], "coverage": [coverage]}
                # the contribution is actually our numerator in the weighted average calculation.
                # we add the weight times how much of the second was in the window, times the probability
                dict_of_sec_vals[second]["prob"].append(window_prob)
                # the contribution is actually our denominator in the weighted average calculation.
                # we add the weight times how much of the second was in the window
                dict_of_sec_vals[second]["coverage"].append(coverage)
        # we go over the dict keys which is the seconds of the recording
        for second in dict_of_sec_vals.keys():
            label = 0
            # if the second was indeed part of the handwashing period, the label changes to 1
            if second in handwashing_times:
                label = 1
            list_of_probabilities = []
            for i in range(len(dict_of_sec_vals[second])):
                if dict_of_sec_vals[second]["coverage"][i] == 1:
                    list_of_probabilities.append(dict_of_sec_vals[second]["prob"][i])
                else:
                    for j in range(i,len(dict_of_sec_vals)):
                        if (dict_of_sec_vals[second]["coverage"][i] + dict_of_sec_vals[second]["coverage"][j]) == 1:
                            list_of_probabilities.append(dict_of_sec_vals[second]["coverage"][i]*dict_of_sec_vals[second]["prob"][i] + dict_of_sec_vals[second]["coverage"][j]*dict_of_sec_vals[second]["prob"][j])
                            dict_of_sec_vals[second]["coverage"].pop(dict_of_sec_vals[second]["coverage"][j])
                            dict_of_sec_vals[second]["prob"].pop(dict_of_sec_vals[second]["prob"][j])
            if len(list_of_probabilities) < 4:
                if len(list_of_probabilities) > 0:
                    # חישוב הממוצע של האיברים הקיימים
                    mean_val = sum(list_of_probabilities) / len(list_of_probabilities)

                    # חישוב כמה איברים חסרים
                    padding_size = 4 - len(list_of_probabilities)

                    # הוספת הממוצע כמספר הפעמים שחסר
                    list_of_probabilities.extend([mean_val] * padding_size)
                else:
                    # in case the list is completely empty
                    list_of_probabilities = [0.0] * 4
            elif len(list_of_probabilities) > 4:
                raise ValueError ("more windows than expected")



            # seconds_df.append({"recording_identifier": recording,"second": second, "weighted_prob": weighted_prob, "label": label, 'Group number': row["Group number"]})
            seconds_df.append({"recording_identifier": recording, "second": second, "prob_1": list_of_probabilities[0],"prob_2": list_of_probabilities[1], "prob_3": list_of_probabilities[2], "prob_4": list_of_probabilities[3],"label": label,'Group number': recording_data["c"]})
    # we obtain the seconds df and second target, and return it
    seconds_df = pd.DataFrame(seconds_df)
    seconds_target = seconds_df['label']
    seconds_df = seconds_df.drop(columns=['label'])
    return seconds_df, seconds_target

def logistic_regression_for_second_classification (seconds_df, target):
    x = seconds_df[["prob_1", "prob_2", "prob_3", "prob_4"]].copy()
    y = target
    group_indicator = second_df ['Group number']
    best_param = find_best_hp_logistic_regression(x, y, split_name = "Split 2", split_by_group_flag=True, group_indicator=group_indicator,
                                    wrapper_text='')
    logistic_model = train_logistic_regression(x, y, best_hp, time_df = None, split_by_group_flag=True, group_indicator=group_indicator)
    return logistic_model











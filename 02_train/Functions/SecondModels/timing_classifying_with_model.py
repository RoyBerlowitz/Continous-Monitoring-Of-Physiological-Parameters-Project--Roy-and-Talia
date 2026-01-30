import pandas as pd
import math
import numpy as np

from Functions_part_c.window_timing_translator_preprocessing import calculate_window_times
from .timing_classifying_without_model import calculate_time_point_weights
from .markov_model import prepare_data_for_hmm,train_supervised_hmm
from .logistic_regression_model import train_logistic_regression, find_best_hp_logistic_regression

from .choose_thresholds import get_threshold_median, get_absolute_threshold_raw, print_metrics_table
from sklearn.metrics import f1_score, precision_recall_curve
from scipy.ndimage import median_filter
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed
from sklearn.model_selection import  StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, average_precision_score

def translate_prediction_into_time_point_prediction_for_model (windows_df, weight_flag = None ):
    # This Function is meant to switch from windows to seconds, by taking into consideration overlapping.
    # weight_flag is None as we do not weight - we keep each probability seperated.
    # we start by obtaining all the unique recording identifier, as this will be done on each identifier separately
    recordings =  windows_df['recording_identifier'].unique()
    # we will create a new df which is based on seconds instead of windows
    seconds_df = []
    # we iterate over all the recording
    for recording in recordings:
        # we obtain each recording data
        recording_data = windows_df[windows_df['recording_identifier'] == recording]

        dupes = recording_data.duplicated(subset=['First second of the activity']).sum()
        if dupes > 0:
            print(f"🚨 Recording {recording} has {dupes} duplicate windows!")
            # פתרון זמני:
            recording_data = recording_data.drop_duplicates(subset=['First second of the activity'])

        # we go over all the complete seconds from the start to end of each recording
        # recording_seconds = range(int(math.floor(recording_data['window_starting_point'].min())), int(math.floor(recording_data['window_ending_point'].max() + 1)))
        recording_seconds = range(int(math.floor(recording_data['First second of the activity'].min())), int(math.floor(recording_data['Last second of the activity'].max() + 1)))
        # creates a dict whose keys are the seconds and the assigned values are None
        dict_of_sec_vals = dict.fromkeys(recording_seconds)
        # we get the exact seconds in which handwashing was operated
        handwashing_times = recording_data["Handwashing time"].iloc[0]

        # we iterate over every row of the recording data, which in that context is a window
        for index, row in recording_data.iterrows():
            # we find the information for each second in the window
            window_time_list = calculate_time_point_weights(row["window_times"], row['First second of the activity'], row['Last second of the activity'], weights_method = weight_flag)
            # we get the probability which was already predicted by model
            window_prob = row["window_probability"]
            # we extract the data of every second.
            # we take the probability and the coverage - what percent of the second the window covers
            for sec_dict in window_time_list:
                second = sec_dict["time_point"]

                coverage = sec_dict["coverage"]
                # if this is the first time we see this second - we create the dict
                if dict_of_sec_vals[second] is None:
                    dict_of_sec_vals[second] = {"prob": [window_prob], "coverage": [coverage]}
                # we save the probability from each window and in adjacent location we save the coverage.
                else:
                    dict_of_sec_vals[second]["prob"].append(window_prob)
                    dict_of_sec_vals[second]["coverage"].append(coverage)

        # we go over the dict keys which is the seconds of the recording
        for second in dict_of_sec_vals.keys():
            label = 0
            # if the second was indeed part of the handwashing period, the label changes to 1
            if second in handwashing_times:
                label = 1
            list_of_probabilities = get_4_probs(dict_of_sec_vals[second]["coverage"], dict_of_sec_vals[second]["prob"], second, recording)

            # list_of_probabilities = []
            # # we iterate over the probabilities from each window
            # for i in range(len(dict_of_sec_vals[second])):
            #     # if the coverage is one - we just take the prob
            #     if dict_of_sec_vals[second]["coverage"][i] == 1:
            #         list_of_probabilities.append(dict_of_sec_vals[second]["prob"][i])
            #     else:
            #         # if the coverage is not one, we find its complements and calculate a weighted average of their probabilities.
            #         for j in range(i,len(dict_of_sec_vals)):
            #             if (dict_of_sec_vals[second]["coverage"][i] + dict_of_sec_vals[second]["coverage"][j]) == 1:
            #                 list_of_probabilities.append(dict_of_sec_vals[second]["coverage"][i]*dict_of_sec_vals[second]["prob"][i] + dict_of_sec_vals[second]["coverage"][j]*dict_of_sec_vals[second]["prob"][j])
            #                 # we remove the complementary to avoid double weighting
            #                 dict_of_sec_vals[second]["coverage"].pop(dict_of_sec_vals[second]["coverage"][j])
            #                 dict_of_sec_vals[second]["prob"].pop(dict_of_sec_vals[second]["prob"][j])
            # # we have 25% overlap, so each time point, apart from the edges, should have 4 probabilities.
            # # for the edges in which there are less prob, we take the average
            # if len(list_of_probabilities) < 4:
            #     if len(list_of_probabilities) > 0:
            #         # we calcualte the mean prob of the second
            #         mean_val = sum(list_of_probabilities) / len(list_of_probabilities)
            #
            #         # we calculate how many missing probs
            #         padding_size = 4 - len(list_of_probabilities)
            #
            #         # we add the padding
            #         list_of_probabilities.extend([mean_val] * padding_size)
            #     else:
            #         # in case the list is completely empty
            #         list_of_probabilities = [0.0] * 4
            # # if there are more than 4 - something not working - we raise value error
            if len(list_of_probabilities) > 4:
                raise ValueError (f"more windows than expected for second{second} in recording {recordings}")
            # seconds_df.append({"recording_identifier": recording,"second": second, "weighted_prob": weighted_prob, "label": label, 'Group number': row["Group number"]})
            seconds_df.append({"recording_identifier": recording, "second": second, "prob_1": list_of_probabilities[0],"prob_2": list_of_probabilities[1], "prob_3": list_of_probabilities[2], "prob_4": list_of_probabilities[3],"label": label,'Group number': recording_data["c"].iloc[0]})
    # we obtain the seconds df and second target, and return it
    seconds_df = pd.DataFrame(seconds_df)
    seconds_target = seconds_df['label']
    seconds_df = seconds_df.drop(columns=['label'])
    return seconds_df, seconds_target

def nearest_valid(c):
    VALID_COVERAGES = np.array([1.0, 0.75, 0.5, 0.25])
    # if c not in VALID_COVERAGES:
    #     print(f"invalid coverage {c}")
    return VALID_COVERAGES[np.argmin(np.abs(VALID_COVERAGES - c))]

def get_4_probs(coverage, probs, sec, recording):
    # there are diff possibilities of coverage for each second. ie [1, 0.25],[1, 1, 1, 0.75] ,[0.75, 1, 1, 1, 0.25],[1, 1, 1]
    # we are aiming to get 4 probabilities for each second.
        # if the coverage is [1,1,1,1] then each probability is used
        # if coverage is [0.25,1,1,1,0.75] then the probabilities of 1 are used and the weighted sum of 0.25,0.75
        # the rest of the cases are only for signal edge and not relevant to many seconds (abt 5secs in beginning and 5 in end)
        # other cases are whatever is coverage 1 will be used and to complete the set of 4 probs then the rest will be padded using a weighted mean of all probs
        # additionally, if something has a coverage that is not 1,0.25,0.5,0,75 (bc end of recording didnt have enough) we will round to these nums

    coverage = np.array([nearest_valid(c) for c in coverage])
    probs = np.array(probs)

    # Case A: perfect coverage
    if len(coverage) == 4 and np.all(coverage == 1):
        return probs.tolist()

    # Split full / partial
    full_mask = (coverage == 1.0)
    partial_mask = ~full_mask

    full_probs = probs[full_mask]
    result = list(full_probs)
    partial_probs = probs[partial_mask]
    partial_cov = coverage[partial_mask]
    # [0.75, 1, 1, 1, 0.25] or [0.5, 1, 1, 1, 0.5]
    if np.any(partial_mask):
        weighted_partial = np.sum(partial_probs * partial_cov) / np.sum(partial_cov)
        result.insert(0,weighted_partial)
        if len(result) == 4:
            return result
        elif len(result) < 4:
            # Weighted mean of partials if exist
            if len(partial_probs) > 0:
                weighted_partial = np.sum(coverage * probs) / np.sum(coverage)
            else:
                weighted_partial = None

            # Fill until 4
            while len(result) < 4:
                if weighted_partial is not None:
                    result.append(weighted_partial)
                else:
                    # fallback: mean of all
                    result.append(np.mean(probs))

    # if len(coverage) == 5:
    #     result.insert(0, coverage[0]*probs[0] + coverage[-1]*probs[-1])
    #     return result

    if len(coverage) > 4:
        print(f'ERROR coverage more that 4 ${coverage} - {len(coverage)} sec {sec} recording {recording}')

    if len(partial_probs) > 0:
        weighted_partial = np.sum(coverage * probs) / np.sum(coverage)
    else:
        weighted_partial = None

    # Fill until 4
    while len(result) < 4:
        if weighted_partial is not None:
            result.append(weighted_partial)
        else:
            # fallback: mean of all
            result.append(np.mean(probs))

    return result[:4]

def logistic_regression_for_second_classification (seconds_df, y):
    # we train a logistic regression for classifying per seconds.
    # the pipeline is the same as logistic regression in part_b
    # we take the relevant columns for train and eval
    x = seconds_df[["prob_1", "prob_2", "prob_3", "prob_4"]].copy()
    # we define the group indicator to be the group number
    group_indicator = seconds_df['Group number']
    # we find the best hyperparameters
    best_param = find_best_hp_logistic_regression(x, y, split_name = "Split 2", split_by_group_flag=True, group_indicator=group_indicator,
                                    wrapper_text='_wrapper')
    # we train the model and find the optimal threshold
    logistic_model = train_logistic_regression(x, y, best_param, time_df = None, split_by_group_flag=True, group_indicator=group_indicator)
    return logistic_model


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve


def train_markov_model(seconds_df, target, n_splits=5):
    # we train the markovian model
    # we start by preparing the data
    group_indicator = seconds_df['Group number']
    X_full, y_full, lengths_full = prepare_data_for_hmm(seconds_df, target)

    # Here, we try to use the power of the PRC curve to find the best operating point in regard of F1.
    # we face a challenge - we try to estimate the PRC without overfitting, which is non-trivial based on the fact we find the best operating point with the data we trained on.
    # we use the Cross-validation prediction - we train again but in a 5-folds scheme, so we get each time the probabilities on data the model "did not see".
    # we use the result to predict the probabilities and by that find the best operating point.
    # it is not exactly the same model, but it is close and justified estimation.
    # we preserve the same logic regarding the group k-folds also here
    # we use stratified groups k-fold to resemble the challenge the model faces
    cv = StratifiedGroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(y_full))
    print(f"Starting Manual CV for HMM with {n_splits} folds...")

    # here we create the cv lopp
    for train_idx, val_idx in cv.split(seconds_df, target, groups=group_indicator):
        # we get the datasets for this fold
        df_train = seconds_df.iloc[train_idx]
        df_val = seconds_df.iloc[val_idx]
        y_train = target.iloc[train_idx]
        y_val = target.iloc[val_idx]

        # we prepare the data for this fold, both train and validatiom
        X_train_fold, y_train_fold, lengths_train_fold = prepare_data_for_hmm(df_train, y_train)
        X_val_fold, y_val_fold, lengths_val_fold = prepare_data_for_hmm(df_val, y_val)

        # we train the hidden markov model on the folds data
        fold_model = train_supervised_hmm(X_train_fold, y_train_fold, lengths_train_fold)

        # we predict the probabilities of the model
        probs = fold_model.predict_proba(X_val_fold, lengths_val_fold)
        oof_probs[val_idx] = probs[:, 1]

    # we find the optimal threshold in terms of maximizing F1 score based on the cross-validation predictions - unbiased data
    precisions, recalls, thresholds = precision_recall_curve(y_full, oof_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[min(best_idx, len(thresholds) - 1)]

    # we train the model on the entire data
    final_model = train_supervised_hmm(X_full, y_full, lengths_full)
    # we declare the optimal threshold
    final_model.optimal_threshold_PRC_ = optimal_threshold

    print(f"Final HMM Optimal Threshold: {optimal_threshold:.4f}")
    return final_model






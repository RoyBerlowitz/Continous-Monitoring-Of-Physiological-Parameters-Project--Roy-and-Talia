import pandas as pd
import numpy as np
import math
from Functions_part_c.window_timing_translator_preprocessing import calculate_window_times
from .choose_thresholds import get_threshold_median, get_absolute_threshold_raw, print_metrics_table
from sklearn.metrics import f1_score, precision_recall_curve
from scipy.ndimage import median_filter
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed


def calculate_time_point_weights (window_times, window_starting_point, window_ending_point, weights_method, overlap = 0.25):
    # we are using the predicted probabilities of the model to predict and classify the label of each second, considering all the windows the seconds is contained in.
    # however, we want to weight the second differently if it is closer to the midpoint, as in theory it tells more about the "essence" of the window.
    # We also want to understand how the second one is included in the window, whether only part of it and if so, weighted accordingly.
    # we receive as an input the window times - the seconds in the window, the start and end points.
    # we also receive the overlap between window - which is 75%, meaning the jump is every 25% of the window duration
    # the weights_method method determine if we weight by a gaussian manner or by triangular manner
    # we start by computing the window duration
    window_duration = window_ending_point - window_starting_point
    # we extract the mid point
    mean_time_point = 0.5 * (window_starting_point + window_ending_point)
    # we create a list that will hold dict of the second, its weight and how many part of it is the window
    time_list = []
    # we iterate over the time point of the window. our rule is that the second is defined by its inferior limit (1-2 sec is the sec 1)
    for i in range(0,len(window_times)-1):
            # we create the dict for each time point
            time_dict = {}
            time_dict ["time_point"] = math.floor(window_times[i])
            time_dict ["coverage"] = window_times[i+1] - window_times[i]

            # we take the floor according to the pre-described convention
            second_start = math.floor(window_times[i])
            second_end = second_start + 1
            # we conduct a check to make sure the window that start in x-7 won't count as a window for sec x
            overlap = max(
                0.0,
                min(window_times[i + 1], second_end) -
                max(window_times[i], second_start)
            )

            if overlap == 0:
                continue

            time_dict["time_point"] = second_start
            time_dict["coverage"] = overlap
            # the weighting will be done based on the distance between the  mid point of 'narrower window' and the mid point of the window
            mid_point_time = 0.5*(window_times[i] + window_times[i+1])
            mid_point_distance_from_mean = np.abs(mean_time_point - mid_point_time)
            if weights_method == "Triangular Weight":
                # if we use the triangular weighting
                weight = 1- mid_point_distance_from_mean/ (0.5 * window_duration)
            elif weights_method == "Gaussian Weight":
                # if we use the gaussian weighting
                # we choose sigma to be the difference between adjacent windows
                weight = np.exp(-(mid_point_distance_from_mean)**2/ (2*((overlap*window_duration)**2)))
            # if the we do not want any weighting
            elif weights_method is None:
                weight = 1
            # for each second dict we save the computed weight and how much of a second it is
            time_dict["weight"] = weight
            time_list.append(time_dict)
    return time_list


def translate_prediction_into_time_point_prediction_with_weights (windows_df, weight_flag = "Gaussian Weight" ):
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
                    dict_of_sec_vals[second] = {"contribution": 0, "weight": 0}
                # the contribution is actually our numerator in the weighted average calculation.
                # we add the weight times how much of the second was in the window, times the probability
                dict_of_sec_vals[second]["contribution"] += weight * coverage * window_prob
                # the contribution is actually our denominator in the weighted average calculation.
                # we add the weight times how much of the second was in the window
                dict_of_sec_vals[second]["weight"] += weight * coverage
        # we go over the dict keys which is the seconds of the recording
        for second in dict_of_sec_vals.keys():
            label = 0
            # if the second was indeed part of the handwashing period, the label changes to 1
            if second in handwashing_times:
                label = 1
                # if we do not have information or window in the sec - we classify as zero, thus zero weight
            if (second not in dict_of_sec_vals) or (not dict_of_sec_vals[second]) or (dict_of_sec_vals[second]["weight"] == 0):
                weighted_prob = 0
            else:
                # for each second, we calculate the weighted average probabillity
                weighted_prob = dict_of_sec_vals[second]["contribution"] / dict_of_sec_vals[second]["weight"]
            # we add the data for each second
            # seconds_df.append({"recording_identifier": recording,"second": second, "weighted_prob": weighted_prob, "label": label, 'Group number': row["Group number"]})
            seconds_df.append({"recording_identifier": recording, "second": second, "weighted_prob": weighted_prob, "label": label,'Group number': recording_data["c"]})
    # we obtain the seconds df and second target, and return it
    seconds_df = pd.DataFrame(seconds_df)
    seconds_target = seconds_df['label']
    seconds_df = seconds_df.drop(columns=['label'])
    return seconds_df, seconds_target

def train_for_decision (X_sec, y_sec, group_indicator, n_iteration =50, n_jobs = -1 ):
    # this function is intended to help find the optimal threshold, the best working point in regard of maximizing the F1 score.
    # as we want to have our model sensitive to the minority group but with high precision. F1 score takes into account both.
    print("Started choosing thresholds...")
    #groups = group_indicator
    # we get the probabilities - these are from the cross validation kfolds done in the train
    y_probs = X_sec['weighted_prob'].values

    # Note: the probabilities which were taken from the model for this part, was the probabilites which were extracted via cross validation on the validation set.
    # in this cross validation process, every model was trained on different data, so they represent prob on unfamiliar data.
    # Thus, they should represent well a similiar task to the test.

    # we get the best threshold - before applying median filtering
    best_raw_threshold = get_absolute_threshold_raw(y_sec,y_probs)
    y_pred_raw_threshold = (y_probs >= best_raw_threshold).astype(int)

    print("\n" + "=" * 40)
    print("      RAW THRESHOLD RESULTS      ")
    print("=" * 40)
    print(f"Best Threshold:   {best_raw_threshold:.4f}")
    print(f"Best F1 Score:    {f1_score(y_sec, y_pred_raw_threshold, zero_division=0):.4f}")
    print("=" * 40)

    no_smoothing_stats = print_metrics_table(y_sec, y_pred_raw_threshold, "Metrics Table For Chosen Threshold Before Median Filtering TRAIN")

    # we find the best threshold after applying median filter
    # we iterate by random search over wide variety of possibilites, and between two option of filter size:
    # 3 takes only the closet neighbors, which is more accurate but less noise filtering
    # 5 takes the two nearest neighbors which filters noise more but less accuracte
    # 7 and more is the size of window so it is irrelevant
    random_thresholds = np.random.uniform(0.1, 0.9, size=n_iteration)
    random_filter_sizes = np.random.choice([3, 5], size=n_iteration)

    # we run it in parallel so the it runs faster
    median_results = Parallel(n_jobs=n_jobs)(
        delayed(get_threshold_median)(X_sec, y_probs, y_sec, filter_size, threshold)
        for threshold, filter_size in zip(random_thresholds, random_filter_sizes)
    )
    # we find the best filter from the random search after median filtering
    best_idx = np.argmax(median_results)
    best_f1_median_threshold = median_results[best_idx]
    best_median_threshold = random_thresholds[best_idx]
    best_filter_size = random_filter_sizes[best_idx]
    y_pred_after_median_filter = (y_probs >= best_median_threshold).astype(int)

    print("\n" + "=" * 40)
    print("      MEDIAN SEARCH RESULTS      ")
    print("=" * 40)
    print(f"Best Filter Size: {best_filter_size}")
    print(f"Best Threshold:   {best_median_threshold:.4f}")
    print(f"Best F1 Score:    {best_f1_median_threshold:.4f}")
    print("=" * 40)


    with_smoothing_stats = print_metrics_table(y_sec, y_pred_after_median_filter, "Metrics Table For Chosen Threshold After Median Filtering TRAIN")

    # we return both thresholds
    return best_raw_threshold, best_median_threshold, best_filter_size, {'train_no_smoothing': no_smoothing_stats, 'train_with_smoothing': with_smoothing_stats}


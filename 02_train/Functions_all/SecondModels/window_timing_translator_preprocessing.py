import pandas as pd
import numpy as np
import math
from scipy.ndimage import median_filter



def create_df_for_time_classification(X_selected):
    # we create a df for the per-second classification
    df_for_timing_classification = X_selected[["First second of the activity", "Last second of the activity"]].copy()
    # we create an identifier to help us distinguish between different recordings - as in this task it is important
    df_for_timing_classification["recording_identifier"] = X_selected['Group number'] + "_" + X_selected['Recording number'] + "_" + X_selected['Participant ID']
    # we save the group number for the latter group k-folds
    df_for_timing_classification["c"] = X_selected['Group number']
    # we find the time points of each window
    df_for_timing_classification["window_times"] = (
        df_for_timing_classification.apply(
            lambda row: calculate_window_times(
                row["First second of the activity"],
                row["Last second of the activity"]
            ),
            axis=1
        )
    )
    # we return the dataframe for further use
    return df_for_timing_classification

def get_handwashing_times (df_for_timing_classification,data_files):
    # we get the raw data, before the segmentation to windows, to be able to locate the seconds in which handwashing was done
    # the keys are the recording identifiers
    for key in data_files.keys():
        # we initiate a list of the handwashing times
        data_files[key]['Handwashing times'] = []
        # this leads us to the data itself
        recording = data_files[key]['label']['data']
        # we get the phases in which handwashing was operated
        relevant_phases = recording[(recording["Label".upper()] == 1)]
        # we iterate over the handwashing periods found
        for index, row in relevant_phases.iterrows():
            # we extract the times of handwashing activities, and also see the Mean and Std to effectively declare windows
            start_col = 'Start (Seconds from Recording Start)'.upper()
            end_col = 'End (Seconds from Recording Start)'.upper()
            # we take them as an int
            start = int(row[start_col])
            end = int(row[end_col])
            # we add to a list of seconds of handwashing the seconds in which handwashing was done, as the range of integers between start and end
            data_files[key]['Handwashing times'] += range(start, end+1)

    # adding for each row of a specific recording in the df of the classification time the handwashing seconds
    valid_keys = set(df_for_timing_classification["recording_identifier"])
    # we iterate over the recording which is in the df we operate on - as the sepration to train and test was already done,
    # each data frame contain partial information
    for key in valid_keys & data_files.keys():
        # We are adding a column of a handwashing list that describes the seconds during which a handwash occurred.
        # The column has the same value for every recording, and the use of this configuration will be explained later.
        # df_for_timing_classification.loc[
        #     df_for_timing_classification["recording_identifier"] == key,
        #     "Handwashing time"
        # ] = data_files[key]["Handwashing times"]
        mask = df_for_timing_classification["recording_identifier"] == key
        times = data_files[key]["Handwashing times"]
        df_for_timing_classification.loc[mask, "Handwashing time"] = pd.Series(
            [times] * mask.sum(),
            index=df_for_timing_classification.index[mask],
            dtype="object"
        )

    # we return the data frame with the handwashing times
    return df_for_timing_classification

def calculate_window_times (window_starting_point, window_ending_point):
    # we calculate the times between the starting and ending point of the window.
    # we get every complete second between them and the first and last

    # there was an issue of getting  [0.0, 0, 1, 2, 3, 4, 5, 6, 7, 7.0] so we added the set to eliminate duplicated
    window_times = sorted(set([window_starting_point] + list(range(math.ceil(window_starting_point), math.floor(window_ending_point) + 1)) + [window_ending_point]))
    return window_times

def apply_smoothing(df, window_size=3):
    # we apply a median filter smoothing,
    # in order to prevent cases in which all of the neighbors are of one label and there is a mid second with distinct label, which is obviously a mislabel.
    # the window size determine the number of neighbors we look at - and the default 3 means if the sample before and after point on another label,
    # we shall label it.
    # the mode constant and c_val = 0 tells him to add padding of zeros in the edges - as in the projects it was defined that we do not start to handwash immediately
    df['smoothed_prediction'] = df.groupby('recording_identifier')['prediction'].transform(
        lambda x: median_filter(x, size=window_size, mode='constant', cval=0)
    )
    return df

def create_test_time_df(X_test, model, selected_feats):
    df_for_time_classification = create_df_for_time_classification(X_test)
    y_prob = model.predict_proba(X_test[selected_feats])[:, 1]
    df_for_time_classification["window_probability"] = y_prob

    return df_for_time_classification




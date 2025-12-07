#-------Part D: Feature Correlation Analysis -------
from Functions import load_data
from Functions.segment_signal import segment_signal
from Functions.extract_features import extract_features
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from npeet_plus import mi
import time



#from caimcaim import CAIMD

import pandas as pd


# data_path = r"C:\Users\nirei\PycharmProjects\Continous monitoring\data"

"""
As we were not sure what is the best way to divide the windows, and the full data from all participants was not yet available - 
we decided to make some kind of heuristic check on our data to identify the candidates for the ideal time for the window.
It should be noted that we left in # comment all the checks and runs we have done, for the checker to understand the entire process.

We started by dividing the potential time periods into 3 ranges -

1) Long (8 - 20 seconds, with 1 second differences)
2) Short (0.2 - 1 second, with 0.2 second differences)
3) Medium (1.5-7.5 seconds, with 0.5 second differences)

for each potential time window, there was a check of 3 overlap - 25%, 50% and 75%.

This test was made firstly by average KSG mutual information between the label to the combined space of skewness and kurtosis of the signal magnitude of Accelerometer and Gyroscope.
It was operated on the data of our group.
The idea was to estimate which lengths are the best, in order to isolate the better preforming ones before commiting the check on the entire data.
The entire data is much larger so we needed to concentrate to less options in order to reduce complexity.

This heuristic check raised the conclusion that longer windows are better in case of the skewness+kurtosis space to label MI.
the short one was ineffective while valid MI (of 4%) was achieved through long windows (17-20 seconds).
However, there were 3 problem - Firstly, our data consisted of longer Hand washing periods in comparison to other groups, what may have affected it. 
Secondly, long windows are not ideal in order to deal with the task of identifying the exact seconds of Handwashing. 
Lastly, Kurtosis and Skewness are metric that may operate better with entire movements and not fractions, so it makes sense that longer windows that allows capturing the whole movement are best.
but other features may not benefit from the same conclusions.

So afterwards, we conducted the same check on Frequency-domain features, with taking the MI of them - in order to see how well their space operates with different window lengths.
This resulted in a much better MI for all the examined window, with no definitive result regarding the size of the window.
However, a clear preference for 75% overlap was observed (represented here by 0.25 since this is multiplied by the window size to indicate the next starting point)

We saw that there is less difference in the MI after the rise from 14 to 20 seconds window, and also that window with length less than 5% operates in a much less effectiveness.
So we conducted a check over the lengths of 5 to 14 seconds, with overlap of 75%.

We got that the longer window operates better. 

However, for the task of classification of the specific times in which the Handwashing took place, not only the correlation is important but also the resolution.
Thus, we decided to choose window of duration = 7 seconds, which is not far after the best windows, but also will allow quite good resolution in afterwards classification.
"""


def feature_correlation(X_features, Y_vector, case = "distribution"):
    #We looked for the best window, with the criterion we defined being the strongest correlation between the features in the paper and the label.
        df = pd.DataFrame()
        MI = 0
        #CASE 1: by the connection between MI and the features of the article - Skewness and Kurtosis
        if case == "distribution":
            cols_suffixes = ["kurtosis", "skewness"]
        #CASE 2 As explained above, this section is intended for the search after the best window length using Frequency-domain features.
        elif case == "Frequency":
            cols_suffixes = ['spectral_entropy','total_energy', 'frequency_centroid','dominant_frequency', 'frequency_variance']
        #we will look at the signal magnitude axes as it takes into consideration all the data, and for reducing time reasons
        #we will look at the Acceloremeter and Gyroscope as we believe they are more significant
        list_of_cols = []
        for sensor in ["Acc", "Gyro"]:
            for suffix in cols_suffixes:
                df[sensor + "_SM_" + suffix] = X_features[sensor + "_SM_" + suffix]
                # As mutual information (MI) is more stable, we chose to find the MI of the columns
                # However, that require discretization.
                # the task of choosing is not trivial, as the number of bins is unclear, and the more advanced methods presents hardness
                # as we didn't want to use algorithm that try to optimize based on groups in order to prevent data leakage due to that.
                # After review, we found Kraskov MI estimator (KSG), which is MI estimator which can done on continous data.
                # It requires Z normalization - and that what we will do
                df[sensor + "_SM_" + suffix] = (df[sensor + "_SM_" + suffix] - df[sensor + "_SM_" + suffix].mean()) / df[sensor + "_SM_" + suffix].std()
                #We add to the list of columns that the MI will be calculated on
                list_of_cols.append(df[sensor + "_SM_" + suffix])
            X = np.column_stack(list_of_cols)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            print("found_MI")
            MI += mi(X, Y_vector, k=7)
        #we'll take the average
        MI = 0.5 * MI
        return MI

# We took 3 sub categories of the window length - short, medium, and long - and try to see which time periods out of the possibilities inside them operates the best
# The idea is to take the best time from those categories and preform more limited search on the entire data based on those times
short_window_duration_options = np.linspace(0.2, 1, 5)
medium_window_duration_options = np.linspace(1.5, 7.5, 13)
long_window_duration_options = np.linspace(8, 20, 13)

def run_single_search(data_path, duration, overlap, case = "distribution"):
    # This function is intended to preform all the required stages for every search.
    # It takes the data path in order to import the data, the duration and over lap that the test is conducted on, and the case in order to differentiate between MI and Relief as a score
    data_files = load_data(data_path)

    # segmentation
    X_matrix, Y_vector = segment_signal(duration, overlap * duration, data_files,more_prints=True)

    # feature extraction
    X_features = extract_features(X_matrix, data_files,more_prints=True)

    # we find the score - based on MI or Relief
    score = feature_correlation(X_features, Y_vector, case)


    return (duration, overlap, score)

def find_best_windows(data_path, window_duration_options, overlap_options, n, case = "distribution"):
    #This function recieves as an input data path which is crucial for the creation of the matrices, the option for window duration, and the number n of n best option we want to take
    # The setup of the function is meant to use all CPU cores and run a parallel search that will accelarate time

    # We wanted to not only check for the best length but also for the best overlap/delay - so for each time we preform the search over 3 possibilities of overlap - 25%, 50% and 75%
    tasks = [
        (duration, overlap)
        for duration in window_duration_options
        for overlap in overlap_options
    ]

    # we get the results by running in parallel
    results = Parallel(n_jobs=2, verbose=10)(
        delayed(run_single_search)(data_path, duration, overlap, case)
        for (duration, overlap) in tasks
    )

    # results is a list of tuples: (duration, overlap, MI)
    #sorting of the results
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

    # # we get the best n results
    top_items = results_sorted[:n]

    #  we turn the results to DF. later, this DF will be exported as an Excel file.
    output_df = pd.DataFrame([
        {"duration": d, "overlap": o, "MI": mi_val}
        for (d, o, mi_val) in results
    ])

    return top_items, output_df

overlap_options = [0.25, 0.5, 0.75]
#We wanted to check how much time it took
start_time = time.time()
#Now we conducted the check over the different possibilities.
#We exported to excel in order to be able to see the results with our eyes - relevant in case of a very similar results or unexpected results, so in this case we can view the entire data
# top_items_long, best_duration_and_overlap_long = find_best_windows(data_path, long_window_duration_options,overlap_options, n = 1)
# best_duration_and_overlap_long.to_excel("best_duration_and_overlap_long.xlsx", index=False)

# top_items_medium, best_duration_and_overlap_medium = find_best_windows(data_path, medium_window_duration_options, overlap_options, n = 3)
# best_duration_and_overlap_medium.to_excel("best_duration_and_overlap_medium.xlsx", index=False)

# top_items_short, best_duration_and_overlap_short = find_best_windows(data_path, short_window_duration_options,overlap_options, n = 2)
# best_duration_and_overlap_short.to_excel("best_duration_and_overlap_short.xlsx", index=False)

#now we try to find the results of which window is the best based on the window lengths, if we are judging base on frequency features.
# We took the medium and long options, as the short discovered to be irrelevant.
new_windows_options = np.concatenate((long_window_duration_options,medium_window_duration_options))
# We estimate the results by their MI score.
# top_frequency_result, full_freq_result  = find_best_windows(data_path, new_windows_options,overlap_options, n = 1, case = "Frequency")
#We export the results to excell sheet in order to be able to see the results clearly.
# full_freq_result.to_excel("best_duration_and_overlap_freq.xlsx", index=False)
# end_time = time.time()
# # we show the total time the code ran
# elapsed_time = end_time - start_time
# print(f"the code ran for {elapsed_time:.2f}  seconds")

overlap_options = [0.25]
new_windows_options = np.linspace(5, 14, 10)

#Here are the final checks on both sets of features to decide on the best window duration

#top_full_data_freq_results, full_data_freq_results  = find_best_windows(data_path, new_windows_options,overlap_options, n = 3, case = "Frequency")
#top_full_data_stat_results, full_data_stat_results  = find_best_windows(data_path, new_windows_options,overlap_options, n = 3, case = "distribution")
#full_data_freq_results.to_excel("whole results - freq_results.xlsx", index=False)
#full_data_stat_results.to_excel("whole results - stat_results.xlsx", index=False)





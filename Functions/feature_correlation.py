#-------Part D: Feature Correlation Analysis -------
from Functions.segment_signal import segment_signal
from Functions.extract_features import extract_features
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from npeet_plus import mi
import time
from skrebate import ReliefF

import pandas as pd


data_path = r"C:\Users\nirei\OneDrive\Desktop\Bachelors Degree - Biomedical Engineering And Neuroscience\Year 4\Semester A\Continuous Monitoring of Physiological Parameters\PythonProject7\02"

"""
As we were not sure what is the best way to divide the windows, and the full data from all participants was not yet available - 
we decided to make some kind of heuristic check on our data to identify the candidates for the ideal time for the window.

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

So afterwards, we conducted the same check on Frequency-domain features, with taking the relief weight of them - in order to see how well their space operates with differnet window lengths.
"""


def feature_correlation(X_features, Y_vector, case = "MI"):
    #CASE 1: by the connection between MI and the features of the article - Skewness and Kurtosis
    #We looked for the best window, with the criterion we defined being the strongest correlation between the features in the paper and the label.
    if case == "MI":
        df = pd.DataFrame()
        MI = 0
        #we will look at the signal magnitude axes as it takes into consideration all the data, and for reducing time reasons
        #we will look at the Acceloremeter and Gyroscope as we believe they are more significant
        for sensor in ["Acc", "Gyro"]:
            df[sensor + "_SM_" + "kurtosis"] = X_features[sensor + "_SM_" + "kurtosis"]
            df[sensor + "_SM_" + "skewness"] = X_features[sensor + "_SM_" + "skewness"]
            # As mutual information (MI) is more stable, we chose to find the MI of the columns
            # However, that require discretization.
            # the task of choosing is not trivial, as the number of bins is unclear, and the more advanced methods presents hardness
            # as we didn't want to use algorithm that try to optimize based on groups in order to prevent data leakage due to that.
            # After review, we found Kraskov MI estimator (KSG), which is MI estimator which can done on continous data.
            # It requires Z normalization - and that what we will do
            df[sensor + "_SM_" + "kurtosis"] = (df[sensor + "_SM_" + "kurtosis"] - df[sensor + "_SM_" + "kurtosis"].mean()) / df[sensor + "_SM_" + "kurtosis"].std()
            df[sensor + "_SM_" + "skewness"] = (df[sensor + "_SM_" + "skewness"] - df[sensor + "_SM_" + "skewness"].mean()) / df[sensor + "_SM_" + "skewness"].std()
            X = np.column_stack([df[sensor + "_SM_" + "kurtosis"], df[sensor + "_SM_" + "skewness"]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            print("found_MI")
            MI += mi(X, Y_vector, k=7)
        #we'll take the average
        MI = 0.5 * MI
        return MI
    elif case == "Frequency":
        feature_suffixes = ['spectral_entropy', 'total_energy', 'frequency_centroid',
                            'dominant_frequency', 'frequency_variance']
        relief_score = 0
        for sensor in ["Acc", "Gyro"]:
            feature_prefix = sensor + "_SM_"
            new_df =  X.groupby([feature_prefix + 'spectral_entropy', feature_prefix + 'total_energy', feature_prefix + 'frequency_centroid',
                            feature_prefix + 'dominant_frequency', feature_prefix + 'frequency_variance'])
            num_features = new_df.shape[1]
            fs = ReliefF(
                n_features_to_select=num_features,
                n_neighbors=10
            )
            fs.fit(new_df.values, Y_vector.values)
            feature_scores = pd.Series(fs.feature_importances_, index=X.columns)
            #sorted_scores = feature_scores.sort_values(ascending=False)
            mean_score = np.mean(feature_scores)
            relief_score += mean_score
        relief_score /= 2
        return relief_score





# We took 3 sub categories of the window length - short, medium, and long - and try to see which time periods out of the possibilities inside them operates the best
# The idea is to take the best time from those categories and preform more limited search on the entire data based on those times
short_window_duration_options = np.linspace(0.2, 1, 5)
medium_window_duration_options = np.linspace(1.5, 7.5, 13)
long_window_duration_options = np.linspace(8, 20, 13)

def run_single_search(data_path, duration, overlap, case = "MI"):
    #This function is intended to preform all the required stages for every search
    # segmentation
    X_matrix, Y_vector = segment_signal(data_path, duration, overlap * duration)

    # feature extraction
    X_features = extract_features(data_path, X_matrix)

    # MI
    MI = feature_correlation(X_features, Y_vector, case)

    return (duration, overlap, MI)

def find_best_windows(data_path, window_duration_options, n, case = "MI"):
    #This function recieves as an input data path which is crucial for the creation of the matrices, the option for window duration, and the number n of n best option we want to take
    # The setup of the function is meant to use all CPU cores and run a parallel search that will aceelarate time

    # We wanted to not only check for the best length but also for the best overlap/delay - so for each time we preform the search over 3 possibilities of overlap - 25%, 50% and 75%
    tasks = [
        (duration, overlap)
        for duration in window_duration_options
        for overlap in [0.25, 0.5, 0.75]
    ]

    # we get the results by running in parallel
    results = Parallel(n_jobs=-1, verbose=10)(
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

#We wanted to check how much time it took
start_time = time.time()
#Now we conducted the check over the different possibilites.
#We exported to excel in order to be able to see the results with our eyes - relevant in case of a very similar results or unexpected results, so in this case we can view the entire data
top_items_long, best_duration_and_overlap_long = find_best_windows(data_path, long_window_duration_options, n = 1)
best_duration_and_overlap_long.to_excel("best_duration_and_overlap_long.xlsx", index=False)

top_items_medium, best_duration_and_overlap_medium = find_best_windows(data_path, medium_window_duration_options, n = 3)
best_duration_and_overlap_medium.to_excel("best_duration_and_overlap_medium.xlsx", index=False)

top_items_short, best_duration_and_overlap_short = find_best_windows(data_path, short_window_duration_options, n = 2)
best_duration_and_overlap_short.to_excel("best_duration_and_overlap_short.xlsx", index=False)

# print("for the short periods:")
# for i, (key, value) in enumerate(top_items_short):
#     print(f"number {i+1} - {key} with MI of {value}")
# print("for the medium periods:")
# for i, (key, value) in enumerate(top_items_medium):
#     print(f"number {i+1} - {key} with MI of {value}")
# print("for the long periods:")
# for i, (key, value) in enumerate(top_items_long):
#     print(f"number {i+1} - {key} with MI of {value}")


#
# new_windows_options = long_window_duration_options + medium_window_duration_options
# top_relief_result, full_relief_result  = find_best_windows(data_path, new_windows_options, n = 5, case = "Frequency")
# full_relief_result.to_excel("best_duration_and_overlap_freq.xlsx", index=False)
# end_time = time.time()
# # we show the total time the code ran
# elapsed_time = end_time - start_time
# print(f"the code ran for {elapsed_time/ (60^2):.2f}  hours")

import numpy as np
from skrebate import ReliefF
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .vet_features_healper_functions import *

#consts
columns_not_to_normalize = ['First second of the activity','Last second of the activity','Participant ID','Group number','Recording number','Protocol']

def feature_normalization(X_train,X_test,method='IQR'):
    features = [col for col in X_train.columns if col not in columns_not_to_normalize]

    #train norm
    norm_train, scaler = normalize_fit(X_train[features].values, method)
    X_train_norm = X_train.copy()
    X_train_norm[features] = norm_train.reshape(-1, len(features)) #this reshape makes the num of rows automatic (-1) and columns len of features. reshapes if flatten happened

    #test norm - uses scaler from the train data set
    norm_test = normalize_transform(X_test[features].values, scaler)
    X_test_norm = X_test.copy()
    X_test_norm[features] = norm_test.reshape(-1, len(features))

    return X_train_norm, X_test_norm

def find_best_features_to_label_combination (X_train, Y_train, administrative_features, more_prints, N=20, K= 10, threshold=0.8):
    #This function tries to find the 20 best features by using a filter method with the CFS metric.
    #It takes as an input X_train, which is the matrix that contains the data and the features
    #It also receives Y_train which gives the labels for each window.
    # N is the number of features we want to find and keep at the matrix
    candidate_columns = X_train.columns
    #We are removing the administrative columns, as we don't want the model to classify based on them - the exact features_names initialized in the vet_features functions.
    candidate_columns = [
        col for col in candidate_columns
        if col not in administrative_features
    ]
    X_vetting = X_train[candidate_columns]
    #To reduce future calculation, we get the feature-to-label correlation matrix and the feature-to-feature correlation matix just once.
    # As Spearman tracks not only linear relationships but also monotonic ones, we choose to calculate it by that
    r_cf = X_vetting.corrwith(Y_train, method='spearman').abs()
    r_ff = X_vetting.corr(method='spearman').abs()
    # We want to keep the best features
    best_features = []
    results_log = []
    if more_prints: print("started searching for best features")
    # As each iteration will add a feature to the best features (we are implementing forward algorithm),
    # we limit the while loop until we find N best features - which mean we have N item in the best_features list
    while len(best_features) < N:
        #We initialize best_score to -inifinity to assure that scores will be added.
        #We don't know the weights values so we can't make any other threshold than that, in case all weights will have lower value - so we choose the most negative number possible
        best_score = -np.inf
        #This variable will be set to the actual chose column
        column_to_add = None
        #We go over all the possible columns
        for column in candidate_columns:
            if more_prints: print(f"iteration {1+len(best_features)}: now checking {column}")
            #for every column, we examine the combination of the already chosen columns/features with the newly tested column
            subset_features = [column] + best_features
            k = len(subset_features)
            # We take the mean feature-to-label correlation for each column we check
            temp_r_cf = r_cf.loc[subset_features].mean()
            if k == 1:
                # If we choose the first feature - there is no correlation with the other features, because there is only one - so we give a zero
                temp_r_ff = 0
            else:
                # For the selected features after the choice of the first one, we get the features corr matrix, and calculate the r_ff according to the method
                temp_r_ff_matrix = r_ff.loc[subset_features, subset_features]
                # We should calculate the mean of the matrix. we should notice that the trace is always 1*k (because the features is always correlated perfectly with itself),
                # So we subtract k from the amount, and then divide it by 2 because we do it hoirozontaly and verticaly (every cor we calculte 2 times)
                temp_r_ff_sum = (temp_r_ff_matrix.sum().sum() - k)/2
                # we get the mean r_ff
                num_pairs = k * (k - 1) / 2
                temp_r_ff = temp_r_ff_sum / num_pairs

            #Now, we finally obtain the CFS score for this combination of features - for the edge case of the first features, it is calculated by the best correlation to label
            CFS_score = (k * temp_r_cf) / np.sqrt(k + k * (k - 1) * temp_r_ff)
            #If this is the best of all the candidate features, we will set the best score to be the score obtained,
            # and the feature to be added will be this feature.
            #if not - nothing happens, and the best so far "survives".
            if CFS_score > best_score:
                best_score = CFS_score
                column_to_add = column
        #We add the chosen feature to the list of best features, and remove it from the candidates - as it was chosen
        best_features.append(column_to_add)
        candidate_columns.remove(column_to_add)
        if more_prints: print(f"Iteration {len(best_features)}:")
        if more_prints: print(f"  Feature Added: {column_to_add}")
        if more_prints: print(f"Current Subset of features: {best_features}")
        if more_prints: print(f"  Current Subset Score: {best_score:.4f}")
        #Now, we get the correlation matrix of test_X.
        #we look for the correlation with the best feature and find the feature which have |0.8| or above correlation with the selected feature.
        #we remove the correlated features, so they will  not be added and we will get only uncorrelated features in the end.
        #We went by spearman correlation as it is less limited to linear connection and catches also monotonic dependency
        full_correlation_matrix = X_vetting.corr(method='spearman')
        correlation_matrix = full_correlation_matrix[column_to_add]
        correlated_features = correlation_matrix[(correlation_matrix.abs() >= threshold)& (correlation_matrix.index != column_to_add)].index.tolist()
        candidate_columns = [
            col for col in candidate_columns
            if col not in correlated_features
        ]
        if more_prints: print(f"features removed: {correlated_features}")
        #the dict is meant in order to be later transformed to excel for interpretability of what happened
        results_log.append({
            "iteration": len(best_features),
            "feature_added": column_to_add,
            "CFS_score": best_score,
            "removed_correlated_features": ", ".join(correlated_features)
        })

        if len(candidate_columns) < N - len(best_features):
            break
    return best_features, results_log


def vet_features_split1(split1, more_prints):
    #each participant was normalized related to its own data. features were vetted for all participants together
    split1_X_trains, split1_X_tests, split1_Y_trains, split1_Y_tests = split1
    all_X_trains = []
    all_X_tests = []
    all_Y_trains = []
    all_Y_tests = []

    for i in range(len(split1_X_trains)):
        X_train, X_test, y_train, y_test = split1_X_trains[i], split1_X_tests[i], split1_Y_trains[i], split1_Y_tests[i]
        # X_train, X_test = vet_features(X_train, X_test, y_train)
        all_X_trains.append(X_train)
        all_X_tests.append(X_test)
        all_Y_trains.append(y_train)
        all_Y_tests.append(y_test)
        # new_dfs.append([X_train, X_test, y_train, y_test])

    all_X_trains = pd.concat(all_X_trains)
    all_X_tests = pd.concat(all_X_tests)
    all_Y_trains = pd.concat(all_Y_trains)

    all_X_vetted, X_test_norm = vet_features(all_X_trains, all_X_tests, all_Y_trains, more_prints)

    return [all_X_vetted, X_test_norm,all_Y_trains,pd.concat(all_Y_tests)]

def vet_features(X_train, X_test, Y_train, more_prints, split_name = "Individual Normalization", N=20, K= 10, threshold=0.8):
    # Preforming the normalization
    X_vetting, X_test_norm = feature_normalization(X_train,X_test, method='IQR')


    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    if "__participant_key__"  in X_train.columns:
        administrative_features.append("__participant_key__")
    # we find the 20 best feature according to CFS and forward algorithm
    best_features, results_log = find_best_features_to_label_combination(X_vetting, Y_train, administrative_features,more_prints, N, K, threshold)
    #we keep the administrative features and most connected to label features, in both train and test
    features_to_keep = administrative_features + best_features
    X_vetting = X_vetting[features_to_keep]
    X_test_norm = X_test_norm[features_to_keep]
    #we try to see the correlation and the pair plot between the features we selected in order to see if we have chosen properly
    if more_prints: print(X_vetting[best_features].corr())

    ##!!!!!!!

    # IF YOU WANT TO SEE THE VETTING PROCESS REMOVE THE COMMENT FROM THE FOLLOWING CODE

    ##!!!!!!!
    #in order to have interpretability for the results, we export to excell with 3 sheets - one with the vetting results, one with the correlation between the features, and one with the pair plot

    # excel_file_path = f"{split_name} vetting data.xlsx"
    # writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
    # pd.DataFrame(results_log).to_excel(
    #     writer,
    #     sheet_name='Vetting Process Log',
    #     index=False
    # )
    # X_vetting[best_features].corr(method = 'spearman').to_excel(
    #     writer,
    #     sheet_name='Correlation Matrix',
    #     index=True)
    # writer.close()

    return X_vetting, X_test_norm


""" 
We first calculated by the relief algorithm but it consumes more memory than the computer can allocate.
We left the function of the selection by relief here, in case before the test we will change the window size and return to this
"""

# def find_best_features_to_label_combination (X_train, Y_train, administrative_features, N=20, K= 10, threshold=0.8):
#     #This function tries to find the 20 best features by using a filter method with the Relief metric.
#     #It takes as an input X_train, which is the matrix that contains the data and the features
#     #It also recieves Y_train which gives the labels for each window.
#     # N is the number of features we want to find and keep at the matrix
#     candidate_columns = X_train.columns
#     #We are removing the administrative columns, as we don't want the model to classify based on them - the exact features_names initialized in the vet_features functions.
#     candidate_columns = [
#         col for col in candidate_columns
#         if col not in administrative_features
#     ]
#     X_vetting = X_train[candidate_columns]
#     # To reduce computational cost
#     X_vetting = X_vetting.astype(np.float32)
#     # We want to keep the best features
#     best_features = []
#     results_log = []
#     print("started searching for best features")
#     # As each iteration will add a feature to the best features (we are implementing forward algorithm),
#     # we limit the while loop until we find N best features - which mean we have N item in the best_features list
#     while len(best_features) < N:
#         #We initialize best_score to -inifinity to assure that scores will be added.
#         #We don't know the weights values so we can't make any other threshold than that, in case all weights will have lower value - so we choose the most negative number possible
#         best_score = -np.inf
#         #This variable will be set to the actual chose column
#         column_to_add = None
#         #We go over all the possible columns
#         for column in candidate_columns:
#             print(f"iteration {1+len(best_features)}: now checking {column}")
#             #for every column, we examine the combination of the already chosen columns/features with the newly tested column
#             test_x = X_vetting[[column] + best_features]
#             #We define the relief algorithm with n features as the number of chosen features + 1 (because we test the combination of them with new column)
#             # n_neighbors means we look for the nth hit and nth miss
#             relief = ReliefF(n_features_to_select=len(best_features) + 1, n_neighbors=K)
#             #we get the relief score for each feature.
#             # The dependency is reflected in the fact that the relief score is calculated based on the distances in the shared feature space of all features.
#             relief.fit(test_x.values, Y_train.values)
#             #the final_score is the sum of all relief weights
#             relief_score = np.sum(relief.feature_importances_)
#             #If this is the best of all the candidate features, we will set the best score to be the score obtained,
#             # and the feature to be added will be this feature.
#             #if not - nothing happens, and the best so far "survives".
#             if relief_score > best_score:
#                 best_score = relief_score
#                 column_to_add = column
#         #We add the chosen feature to the list of best features, and remove it from the candidates - as it was chosen
#         best_features.append(column_to_add)
#         candidate_columns.remove(column_to_add)
#         print(f"Iteration {len(best_features)}:")
#         print(f"  Feature Added: {column_to_add}")
#         print(f"Current Subset of features: {best_features}")
#         print(f"  Current Subset Score: {best_score:.4f}")
#         #Now, we get the correlation matrix of test_X.
#         #we look for the correlation with the best feature and find the feature which have |0.8| or above correlation with the selected feature.
#         #we remove the correlated features, so they will  not be added and we will get only uncorrelated features in the end.
#         #We went by spearman correlation as it is less limited to linear connection and catches also monotonic dependency
#         full_correlation_matrix = X_vetting.corr(method='spearman')
#         correlation_matrix = full_correlation_matrix[column_to_add]
#         correlated_features = correlation_matrix[(correlation_matrix.abs() >= threshold)& (correlation_matrix.index != column_to_add)].index.tolist()
#         candidate_columns = [
#             col for col in candidate_columns
#             if col not in correlated_features
#         ]
#         print(f"features removed: {correlated_features}")
#         #the dict is meant in order to be later transformed to excel for interpratibillity of what happend
#         results_log.append({
#             "iteration": len(best_features),
#             "feature_added": column_to_add,
#             "relief_score": best_score,
#             "removed_correlated_features": ", ".join(correlated_features)
#         })
#
#         if len(candidate_columns) < N - len(best_features):
#             break
#     return best_features, results_log
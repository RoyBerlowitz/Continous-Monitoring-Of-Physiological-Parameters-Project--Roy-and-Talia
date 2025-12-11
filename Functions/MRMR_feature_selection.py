import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def find_relevance_vector (df, label_vector):
    # To reduce the number of calculations, we find the relevance vector in the beginning instead of calculating relevance over and over again.
    # The relevance vector includes the values of the MI of every feature with the label.
    # We start by defining the size of the vector
    relevance_vector = np.zeros(len(df.columns))
    # we go over each feature
    for feature_name in df.columns:
        #we reshape the feature column so it will match the sklearn requirements
        feature = df[feature_name]
        feature = feature.values.reshape(-1, 1)
        #we calculate the feature-label MI and get a np.array of [MI], so we take the 0th item
        MI_score = mutual_info_classif(feature, label_vector.values, discrete_features= True)[0]
        #we add it to the relevance vector in the appropriate position
        relevance_vector[df.columns.get_loc(feature_name)] = MI_score
    return relevance_vector

def find_redundancy_matrix(df):
    # To reduce the number of calculations, we find the redundancy matrix in the beginning instead of calculating redundancy over and over again.
    # The redundancy matrix includes the values of the MI of every pair of features.
    # We start by defining the size of the matrix
    redundancy_matrix = np.zeros((len(df.columns), len(df.columns)))
    n = len(df.columns)
    #we loop over the number of features. we take j < n+1 in order to reduce number of calculation because of the redundancy
    for i in range(0,n):
        for j in range(i+1):
            # we reshape the feature columns so it will match the sklearn requirements
            feature_i = df.iloc[:,i].values.reshape(-1, 1)
            feature_j = df.iloc[:,j].values.reshape(-1, 1)
            # we find the MI score of the pair of features
            MI_score = mutual_info_classif(feature_i, feature_j, discrete_features=True)[0]
            # we locate the MI score in the matching locations of the redundancy matrix
            redundancy_matrix[i, j] = MI_score
            redundancy_matrix[j, i] = MI_score

    return redundancy_matrix

def MRMR_feature_selection (df, label_vector, candidate_columns ,stopping_criteria, more_prints = True):
    # Here, we preform the feature selection according to the MRMR algorirthm.
    # The function takes as an input a df with the feature columns and label vector which is the target.
    # it is given a stopping criteria, whch tells the function to stop adding features, and also candidate columns which are the columns we wish to examine.
    best_features = []
    results_log = []
    #we start by finding the relevance vector and redundancy matrix, instead of computing calculation in each iteration again.
    relevance_vector = find_relevance_vector(df, label_vector)
    redundancy_matrix = find_redundancy_matrix(df)
    # we find the feature with the maximum relevance, and we add him as the first feature/
    max_relevance_index = np.argmax(relevance_vector)
    most_relevant_column = df.columns[max_relevance_index]
    best_features.append(most_relevant_column)
    candidate_columns.remove(most_relevant_column)
    # We go over all the possible columns
    while len(candidate_columns) > 0:
        # We initialize best_score to -infinity to assure that scores will be added - every score is higher.
        best_score = -np.inf
        # This variable will be set to the actual chose column
        column_to_add = None
        for column in candidate_columns:
            if more_prints: print(f"iteration {1 + len(best_features)}: now checking {column}")
            #as we take the relevance and redundancy from the pre-calculated vector and matrix, so we find the index.
            column_index = df.columns.get_loc(column)
            # we take the relevance score from the vector based on the index
            relevance_score = relevance_vector[column_index]
            # we initate the redundancy score as a zero
            subset_redundancy_score = 0
            #we go over the already chose features
            for chosen_feature in best_features:
                # we find the index of the now examined features
                chosen_feature_index = df.columns.get_loc(chosen_feature)
                # we extract the joint redundancy score from the matrix
                redundancy_score = redundancy_matrix[chosen_feature_index, column_index]
                # we add to the total score
                subset_redundancy_score += subset_redundancy_score
            # we find the mean redunndancy score
            subset_redundancy_score /= len(best_features)
            #we compute the total MRMR score of the feature: relevance score - redundancy score
            feature_score = relevance_score - subset_redundancy_score
            # it he MRMR score of the feature is higher than the best score - it becomes the best score
            if feature_score > best_score:
                best_score = feature_score
                # we save the column which gave the best score to be able to add him later.
                # it should be noted that in a case where their is a better feature - it will be changed again
                column_to_add = column
        # The stopping criterion is a minimum threshold for the maximum MRMR score. If it is lower than this value, we assume that the redundancy outweighs the relevance,
        # and we terminate the greedy selection process. The default value will be zero, meaning the process stops when the score is non-positive.
        if best_score < stopping_criteria:
            break
        else:
            #if we didn't stop - we add the feature to selected features, and we remove it from the features we need to check
            best_features.append(column_to_add)
            candidate_columns.remove(column_to_add)
            if more_prints: print(f"Iteration {len(best_features)}:")
            if more_prints: print(f"  Feature Added: {column_to_add}")
            if more_prints: print(f"Current Subset of features: {best_features}")
            if more_prints: print(f"  Current Subset Score: {best_score:.4f}")
            results_log.append({
                "iteration": len(best_features),
                "feature_added": column_to_add,
                "MRMR_score": best_score,
            })

    return best_features, results_log, redundancy_matrix, relevance_vector
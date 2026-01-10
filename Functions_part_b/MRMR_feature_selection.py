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
        labels = label_vector.values.ravel()
        #we calculate the feature-label MI and get a np.array of [MI], so we take the 0th item
        MI_score = mutual_info_classif(feature, labels, discrete_features= True)[0]
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
            feature_j = df.iloc[:,j].values.ravel()
            # we find the MI score of the pair of features
            MI_score = mutual_info_classif(feature_i, feature_j, discrete_features=True)[0]
            # we locate the MI score in the matching locations of the redundancy matrix
            redundancy_matrix[i, j] = MI_score
            redundancy_matrix[j, i] = MI_score

    return redundancy_matrix

def MRMR_feature_selection (df, label_vector, candidate_columns ,stopping_criteria, more_prints = True):
    # Here, we preform the feature selection according to the MRMR algorirthm.
    # The function takes as an input a df with the feature columns and label vector which is the target.
    #  (FIX IT) it is given a stopping criteria, which tells the function to stop adding features, and also candidate columns which are the columns we wish to examine.
    best_features = []
    results_log = []
    #we start by finding the relevance vector and redundancy matrix, instead of computing calculation in each iteration again.
    relevance_vector = find_relevance_vector(df, label_vector)
    redundancy_matrix = find_redundancy_matrix(df)
    # we find the feature with the maximum relevance, and we add him as the first feature/
    max_relevance_index = np.argmax(relevance_vector)
    most_relevant_column = df.columns[max_relevance_index]
    results_log.append({
        "iteration": len(best_features),
        "feature_added": most_relevant_column,
        "MRMR_score": relevance_vector[max_relevance_index],
    })
    best_features.append(most_relevant_column)
    candidate_columns.remove(most_relevant_column)
    # we initiate the values of total relevance and total redundancy that will help us to calculate the MRMR score for the stopping rule.
    # we initiate the MRMR score to be to ensure at least 2 features
    total_relevance = relevance_vector[max_relevance_index]
    total_redundancy = redundancy_matrix[max_relevance_index,max_relevance_index]
    if not total_redundancy == 0:
        MRMR_score = total_relevance/ total_redundancy
    else:
        MRMR_score = 0
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
                subset_redundancy_score += redundancy_score
            # we find the mean redundancy score
            subset_redundancy_score /= len(best_features)
            #we compute the total MRMR score of the feature: relevance score - redundancy score
            feature_score = relevance_score - subset_redundancy_score
            # it the MRMR score of the feature is higher than the best score - it becomes the best score
            if feature_score > best_score:
                best_score = feature_score
                # we save the redundancy and relevance of the chose features
                current_redundancy_score = subset_redundancy_score
                current_relevance_score = relevance_score
                # we save the column which gave the best score to be able to add him later.
                # it should be noted that in a case where there is a better feature - it will change again
                column_to_add = column
                column_to_add_index = df.columns.get_loc(column_to_add)
        # we add to the total relevance the relevance score of the newly added features
        total_relevance += current_relevance_score
        # we add to the redundancy 2*redundancy score (for the new feature and for every former feature with the newly added feature)
        # we also add the MI of the feature with himself
        total_redundancy += 2* current_redundancy_score + redundancy_matrix[column_to_add_index, column_to_add_index]
        # if the new MRMR score is higher than the former one - we continue to add features, as they still contribute significantly.
        # if the MRMR score is falling - it is not significant enough, and we stop adding features
        #if MRMR_score > total_relevance / total_redundancy or best_score < stopping_criteria:
        if best_score <= stopping_criteria:
            break
        else:
            # we update the MRMR score for the next iteration
            MRMR_score = total_relevance / total_redundancy
            # if we didn't stop - we add the feature to selected features, and we remove it from the features we need to check
            best_features.append(column_to_add)
            candidate_columns.remove(column_to_add)
            if more_prints: print(f"Iteration {len(best_features)}:")
            if more_prints: print(f"  Feature Added: {column_to_add}")
            if more_prints: print(f"Current Subset of features: {best_features}")
            if more_prints: print(f"Current Subset Score: {best_score:.4f}")
            results_log.append({
                "iteration": len(best_features),
                "feature_added": column_to_add,
                "MRMR_score": best_score,
            })
        # The stopping criterion is a minimum threshold for the maximum MRMR score. If it is lower than this value, we assume that the redundancy outweighs the relevance,
        # and we terminate the greedy selection process. The default value will be zero, meaning the process stops when the score is non-positive.
        #if best_score < stopping_criteria:

    return best_features, results_log, redundancy_matrix, relevance_vector
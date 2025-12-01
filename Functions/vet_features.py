import numpy as np
from skrebate import ReliefF
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
הערות לעצמי לקראת הביצוע:
- לא לשכוח נורמליזציה
- לשמור כל פעם את העמודות שאני מוריד - לחשוב אם לא פשוט שווה פשוט לעשות דרופ אחרי כל בחירה, ואולי לא לעשות היוריסטית.
- לעשות בסוף איזשהי heat_map של הפיצ'רים שנבחרו
- לא לשכוח לתעד באופן שיהיה קל להבין למה בחרתי מה שבחרתי
- לא לשכוח להוציא את הפיצ'רים ההתחלתיים (קבוצה, זמנים וכאלה) מהבדיקה
- לא לשכוח לשמור את הממוצעים של הנרמול לטובת הפעולה הזו
"""

#def feature_normalization()

def find_best_features_to_label_combination (X_train, Y_train, administrative_features, N=20, K= 10, threshold=0.8):
    #This function tries to find the 20 best features by using a filter method with the Relief metric.
    #It takes as an input X_train, which is the matrix that contains the data and the features
    #It also recieves Y_train which gives the labels for each window.
    # N is the number of features we want to find and keep at the matrix
    candidate_columns = X_train.columns
    #We are removing the administrative columns, as we don't want the model to classify based on them - the exact features_names initialized in the vet_features functions.
    candidate_columns = [
        col for col in candidate_columns
        if col not in administrative_features
    ]
    X_vetting = X_train[candidate_columns]
    # We want to keep the best features
    best_features = []
    results_log = []
    print("started searching for best features")
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
            print(f"iteration {1+len(best_features)}: now checking {column}")
            #for every column, we examine the combination of the already chosen columns/features with the newly tested column
            test_x = X_vetting[[column] + best_features]
            #We define the relief algorithm with n features as the number of chosen features + 1 (because we test the combination of them with new column)
            # n_neighbors means we look for the nth hit and nth miss
            relief = ReliefF(n_features_to_select=len(best_features) + 1, n_neighbors=K)
            #we get the relief score for each feature.
            # The dependency is reflected in the fact that the relief score is calculated based on the distances in the shared feature space of all features.
            relief.fit(test_x.values, Y_train.values)
            #the final_score is the sum of all relief weights
            relief_score = np.sum(relief.feature_importances_)
            #If this is the best of all the candidate features, we will set the best score to be the score obtained,
            # and the feature to be added will be this feature.
            #if not - nothing happens, and the best so far "survives".
            if relief_score > best_score:
                best_score = relief_score
                column_to_add = column
        #We add the chosen feature to the list of best features, and remove it from the candidates - as it was chosen
        best_features.append(column_to_add)
        candidate_columns.remove(column_to_add)
        print(f"Iteration {len(best_features)}:")
        print(f"  Feature Added: {column_to_add}")
        print(f"Current Subset of features: {best_features}")
        print(f"  Current Subset Score: {best_score:.4f}")
        #Now, we get the correlation matrix of test_X.
        #we look for the correlation with the best feature and find the feature which have |0.8| or above correlation with the selected feature.
        #we remove the correlated features, so they will  not be added and we will get only uncorrelated features in the end.
        full_correlation_matrix = X_vetting.corr()
        correlation_matrix = full_correlation_matrix[column_to_add]
        correlated_features = correlation_matrix[(correlation_matrix.abs() >= threshold)& (correlation_matrix.index != column_to_add)].index.tolist()
        candidate_columns = [
            col for col in candidate_columns
            if col not in correlated_features
        ]
        print(f"features removed: {correlated_features}")
        #the dict is meant in order to be later transformed to excel for interpratibillity of what happend
        results_log.append({
            "iteration": len(best_features),
            "feature_added": column_to_add,
            "relief_score": best_score,
            "removed_correlated_features": ", ".join(correlated_features)
        })

        if len(candidate_columns) < N - len(best_features):
            break
    return best_features, results_log


def vet_features(X_train, Y_train, split_name = "Individual Normalization", N=20, K= 10, threshold=0.8):
    X_vetting = copy.deepcopy(X_train) #טליה - צריך להוריד את זה אחרי שעושים את הנורמליזציה
    #X_vetting = add here feature_normalization

    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    if "__participant_key__"  in X_train.columns:
        administrative_features.append("__participant_key__")
    # we find the 20 best feature according to relief and forward algorithm
    best_features, results_log = find_best_features_to_label_combination(X_vetting, Y_train, administrative_features, N, K, threshold)
    #we export the dict that show the different stages to excellfor interpretabillity
    pd.DataFrame(results_log).to_csv(f"{split_name} vetting process.csv", index=False)
    #we keep the administrative features and most connected to label features
    features_to_keep = administrative_features + best_features
    X_vetting = X_vetting[features_to_keep]
    #we try to see the correlation and the pair plot between the features we selected in order to see if we have chosen properly
    print(X_vetting[best_features].corr())
    sns.pairplot(pd.concat([X_vetting[best_features], Y_train.rename("Label")], axis=1), hue="Label")
    plt.show()

    return X_vetting


import copy

import pandas as pd
from Functions_part_b.CACC_discretization import CACC_discretization, discretize_colum
from Functions_part_b.MRMR_feature_selection import MRMR_feature_selection
from joblib import Parallel, delayed


def run_cacc_for_column(df: pd.DataFrame, col_name: str, y_target: pd.Series):
    # This function takes as an input a dataframe and the column name which represent the feature column that should be discretized and a a target series.
    # we first take the values of tje target and column and turn them to be np array
    values = df[col_name].values
    target = y_target.values
    # we find the cut point acording to CACC algorithm
    cut_points = CACC_discretization(values, target, col_name)

    # we discretized the data acordding to the found cut points to be labeled as zero to the number of cut points found - 1
    discretized_series = discretize_colum(df, col_name, cut_points)

    return col_name, discretized_series, cut_points

def select_features(X_vetting, Y_train, X_test, split_name = "Individual_split", stopping_criteria = 0):
    # we create the X_selected
    X_selected = copy.deepcopy(X_vetting)
    # we separate the administrative features from the features we check, to keep them for later use and becasue they won't be part of prediction anyway
    # we create a list of the column which we will conduct the search on them
    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    if "__participant_key__"  in X_vetting.columns:
        administrative_features.append("__participant_key__")
    candidate_columns = X_selected.columns
    candidate_columns =[col for col in candidate_columns if col not in administrative_features]
    # we keep only the relevant feature for the forward search in X_vetting that the check will be conducted on (the administrative feature remained in X_selected)
    X_vetting = X_vetting.drop(columns=administrative_features)
    X_vetting_for_discretization = X_vetting.copy()
    # To use MI, we need to discretize the columns.
    # we do it in a way that won't influence the output X_selected, because we do not want to limit our prediction to small set of discrete values
    # We use parallel calculation as this is the most computationaly complex stage
    discretized_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_cacc_for_column)(X_vetting_for_discretization, col, Y_train)
        for col in candidate_columns
    )
    dict_of_discertization = {}
    #we update the columns
    for col_name, series,cut_points in discretized_results:
        X_vetting[col_name] = series
        dict_of_discertization[col_name] = len(cut_points)

    # for col in candidate_columns:
    #     # To use MI, we need to discretize the columns.
    #     # we do it in a way that won't influence the output X_selected, because we do not want to limit our prediction to small set of discrete values
    #     cut_points= CACC_discretization(X_vetting[col].values, Y_train.values, col)
    #     X_vetting[col] = disceretize_column(X_vetting, col, cut_points)

    # we preform the MRMR feature selection
    best_features, results_log, redundancy_matrix, relevance_vector = MRMR_feature_selection(X_vetting, Y_train, candidate_columns, stopping_criteria, more_prints=True)
    features_to_keep = administrative_features + best_features
    #we let only the administrative and selected features to remain
    X_selected = X_selected[features_to_keep]
    X_test = X_test[features_to_keep]
    #we create excel file for intrepretabillity
    excel_file_path = f"{split_name} feature_selection.xlsx"
    writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
    pd.DataFrame(results_log).to_excel(
        writer,
        sheet_name='Feature Selection Log',
        index=False
    )
    pd.DataFrame(redundancy_matrix).to_excel(
        writer,
        sheet_name='Redundancy Matrix',
        index=True)
    pd.DataFrame(relevance_vector).to_excel(
        writer,
        sheet_name='Relevance Vector',
        index=True)
    pd.Series(dict_of_discertization).to_frame().to_excel(
        writer,
        sheet_name='Discretization process',
        index=True)

    writer.close()

    return X_selected


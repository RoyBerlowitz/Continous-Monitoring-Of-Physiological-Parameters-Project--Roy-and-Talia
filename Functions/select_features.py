import copy

import pandas as pd
import numpy as np
from Functions.CACC_discretization import CACC_discretization
from Functions.MRMR_feature_selection import MRMR_feature_selection

def select_features(X_vetting, Y_train, split_name = "Individual_split", stopping_criteria = 0):
    X_selected = copy.deepcopy(X_vetting)
    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
    if "__participant_key__"  in X_vetting.columns:
        administrative_features.append("__participant_key__")
    candidate_columns = X_selected.columns
    candidate_columns =[col for col in candidate_columns if col not in administrative_features]
    X_vetting = X_vetting.drop(columns=administrative_features)
    for col in candidate_columns:
        X_vetting[col] = CACC_discretization(X_vetting, col)
    best_features, results_log, redundency_matrix, relevance_vector = MRMR_feature_selection(X_vetting, Y_train, candidate_columns, stopping_criteria, more_prints=True)
    features_to_keep = administrative_features + best_features
    X_selected = X_selected[features_to_keep]
    # X_test_norm = X_test_norm[features_to_keep]
    excel_file_path = f"{split_name} feature_selection.xlsx"
    writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
    pd.DataFrame(results_log).to_excel(
        writer,
        sheet_name='Feature Selection Log',
        index=False
    )
    pd.DataFrame(redundency_matrix).to_excel(
        writer,
        sheet_name='Redundancy Matrix',
        index=True)
    pd.DataFrame(relevance_vector).to_excel(
        writer,
        sheet_name='Relevance Vector',
        index=True)

    writer.close()

    return X_selected


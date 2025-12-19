from xgboost import XGBClassifier

from .xgboost_model_helper_functions import xgb_grid_search_multi, xgb_random_search_multi
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import numpy as np
from .evaluate_model_functions import closest_point_roc


def find_best_hp_xgboost(X_train, y_train, split_name, split_by_group_flag = False, group_indicator = None):

    # Grid search
    # best_xgb_grid, best_params_grid, results_grid = xgb_grid_search_multi(X_train, y_train)
    # results_grid.to_excel(f'{split_name}_xgboost_results_xgb_grid.xlsx')
    # print(f'Saved {split_name}_xgboost_results_xgb_grid.xlsx')

    # Randomized search
    best_xgb_rand, best_params_rand, results_rand = xgb_random_search_multi(X_train, y_train, split_by_group_flag=split_by_group_flag, group_indicator=group_indicator)
    results_rand.to_excel(f'{split_name}_xgboost_results_xgb_rand.xlsx')
    print(f'Saved {split_name}_xgboost_results_xgb_rand.xlsx')

    return best_params_rand

def train_xgboost(X_train, y_train, best_hp, random_state=42):

    model = XGBClassifier(
        random_state=random_state,
        eval_metric='logloss',
        **best_hp
    )

    model.fit(X_train, y_train)

    # Here, we try to use the power of the PRC curve to find the best operating point in regard of F1.
    # we face a challenge - we try to estimate the PRC without overfitting, which is non-trivial based on the fact we find the best operating point with the data we trained on.
    # we use the Cross-validation prediction - we train again but in a 5-folds scheme, so we get each time the probabilities on data the model "did not see".
    # we use the result to predict the probabilities and by that find the best operating point.
    # it is not exactly the same model, but it is close and justified estimation.
    y_probs = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    # we calculate the needed calculation for the PRC curve
    precisions, recalls, thresholds = precision_recall_curve(X_train, y_probs)
    avg_prec = average_precision_score(y_train, y_probs)

    # Here we find the optimal threshold, which is the point which gives the best F1 score.
    # F1 score represnt both senstivity and precision and by that hints a lot about the minority group
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    model.optimal_threshold_PRC_ = thresholds[best_idx]

    # We will also find the ROC-AUC optimal point - which is the closet one to the (0,1)
    fpr, tpr, roc_thresholds = roc_curve(y_train, y_probs)
    roc_res = closest_point_roc(fpr, tpr, roc_thresholds)
    model.optimal_threshold_ROC_ = roc_res['threshold']
    return model
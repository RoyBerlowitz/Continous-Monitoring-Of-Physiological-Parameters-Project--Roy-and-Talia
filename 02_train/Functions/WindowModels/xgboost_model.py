from xgboost import XGBClassifier

from .xgboost_model_helper_functions import xgb_grid_search_multi, xgb_random_search_multi, XGBClassifierClassifier
from sklearn.model_selection import cross_val_predict, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ..EvaluateModels.evaluate_model_functions import closest_point_roc


def find_best_hp_xgboost(X_train, y_train, split_by_group_flag = False, group_indicator = None, subsampling_flg = False):
    # Grid search
    # best_xgb_grid, best_params_grid, results_grid = xgb_grid_search_multi(X_train, y_train)
    # results_grid.to_excel(f'xgboost_results_xgb_grid.xlsx')
    # print(f'Saved xgboost_results_xgb_grid.xlsx')

    # Randomized search
    best_xgb_rand, best_params_rand, results_rand = xgb_random_search_multi(X_train, y_train, split_by_group_flag=split_by_group_flag, group_indicator=group_indicator,subsampling_flg = subsampling_flg)
    results_rand.to_excel(f'xgboost_results_xgb_rand.xlsx')
    print(f'Saved xgboost_results_xgb_rand.xlsx')

    return best_params_rand

def train_xgboost(X_train, y_train, best_hp, time_df, random_state=42, split_by_group_flag = False, group_indicator=None):
    # we encode the labels to be ints
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    best_hp_xgb = {
        k.replace('xgb__', ''): v #remove __xgb from param name
        for k, v in best_hp.items()
    }

    model = XGBClassifierClassifier(
        random_state=random_state,
        eval_metric='logloss',
        objective='binary:logistic',
        **best_hp_xgb
    )

    # Here, we try to use the power of the PRC curve to find the best operating point in regard of F1.
    # we face a challenge - we try to estimate the PRC without overfitting, which is non-trivial based on the fact we find the best operating point with the data we trained on.
    # we use the Cross-validation prediction - we train again but in a 5-folds scheme, so we get each time the probabilities on data the model "did not see".
    # we use the result to predict the probabilities and by that find the best operating point.
    # it is not exactly the same model, but it is close and justified estimation.
    # we preserve the same logic regarding the group k-folds also here
    if split_by_group_flag:
        cv_strategy = StratifiedGroupKFold(n_splits=5)
    else:
        cv_strategy = StratifiedKFold(n_splits=5)

    y_probs = cross_val_predict(model, X_train, y_train_encoded, groups=group_indicator, cv=cv_strategy, method='predict_proba')[:, 1]

    model.fit(X_train, y_train_encoded)
    # we calculate the needed calculation for the PRC curve
    precisions, recalls, thresholds = precision_recall_curve(y_train_encoded, y_probs)
    avg_prec = average_precision_score(y_train_encoded, y_probs)

    #we add the calculated probabilities to the df used for obtaining the labeling per second
    time_df["window_probability"] = y_probs

    # Here we find the optimal threshold, which is the point which gives the best F1 score.
    # F1 score represnt both senstivity and precision and by that hints a lot about the minority group
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    model.optimal_threshold_PRC_ = thresholds[min(best_idx, len(thresholds) - 1)]

    # We will also find the ROC-AUC optimal point - which is the closet one to the (0,1)
    fpr, tpr, roc_thresholds = roc_curve(y_train_encoded, y_probs)
    roc_res = closest_point_roc(fpr, tpr, roc_thresholds)
    model.optimal_threshold_ROC_ = roc_res['threshold']

    # if you want to print metrics ================================================================
    # print(model.optimal_threshold_ROC_)
    # y_prob_train = model.predict_proba(X_train)[:, 1]
    # y_pred_cv = (y_probs >= model.optimal_threshold_ROC_).astype(int)
    # y_pred_train = (y_prob_train >= model.optimal_threshold_ROC_).astype(int)
    # print_metrics_table(y_train, y_pred_cv, y_probs, 'Results from train cross-validation')
    # print_metrics_table(y_train, y_pred_train, y_prob_train, 'Results from train on model trained on train')
    # ================================================================================================

    return model


def print_metrics_table(y_true, y_pred, y_prob=None, title="Metrics"):
    # Precision, recall, accuracy, F1, Cohen's kappa
    from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score, \
        cohen_kappa_score, precision_recall_curve, auc, roc_auc_score

    # Basic metrics
    p = precision_score(y_true, y_pred, zero_division=0)
    s = recall_score(y_true, y_pred, zero_division=0)
    a = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # PR AUC
    if y_prob is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        pr_auc = None
        roc_auc = None

    # Print metrics table
    print(f"\n--- {title} ---")
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 28)
    print(f"{'Precision':<15} | {p:.4f}")
    print(f"{'Sensitivity':<15} | {s:.4f}")
    print(f"{'Accuracy':<15} | {a:.4f}")
    print(f"{'F1 Score':<15} | {f1:.4f}")
    print(f"{'Cohen Kappa':<15} | {kappa:.4f}")
    if pr_auc is not None:
        print(f"{'PR AUC':<15} | {pr_auc:.4f}")
    if roc_auc is not None:
        print(f"{'ROC AUC':<15} | {roc_auc:.4f}")
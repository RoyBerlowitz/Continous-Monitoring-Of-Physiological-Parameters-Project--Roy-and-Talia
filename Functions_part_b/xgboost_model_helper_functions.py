from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer, recall_score
from scipy.stats import uniform, randint, loguniform
from xgboost import XGBClassifier
import pandas as pd

result_cols_to_save = [
        'params',
        # TRAIN SCORE
        'mean_train_AUC', 'mean_train_Accuracy', 'mean_train_Specificity', 'mean_train_Sensitivity',
        'mean_train_Precision', 'mean_train_F1', 'mean_train_PRC', 'mean_train_Kappa',
        # TEST SCORES
        'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_Specificity', 'mean_test_Precision',
        'mean_test_Sensitivity', 'mean_test_F1', 'mean_test_PRC', 'mean_test_Kappa',
        # Control columns
        'mean_fit_time', 'rank_test_AUC'
    ]

def xgb_grid_search_multi(X_train, y_train, cv=5):
    """
    XGBoost hyperparameter tuning with multiple scoring metrics using GridSearchCV.

    Returns:
        best_model: trained XGBClassifier
        best_params: dict of best hyperparameters
        results_df: DataFrame with scores for all tested hyperparameter combinations
    """

    kappa_scorer = make_scorer(cohen_kappa_score)

    scoring_metrics = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy',
        'F1': 'f1_macro',
        'Sensitivity': 'recall_macro',
        'PRC': 'average_precision',
        'Kappa': kappa_scorer
    }

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Define a practical hyperparameter grid for XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'scale_pos_weight': [1, pos_weight]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Here we preform the search itself
    # We decided to evaluate our model by the PRC.
    # PRC represents the potential of the model in regard to the positive (which is the minority) group.
    # By finding the point that maximizes the F1 score in the PRC column, we can reach to the pont that hold the best potential F1 score,
    # which may indicate the best balance between sensitivity and precision
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=cv,
        scoring=scoring_metrics,
        refit='average_precision',
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[result_cols_to_save]

    return grid_search.best_estimator_, grid_search.best_params_, results_df

def xgb_random_search_multi(X_train, y_train, split_by_group_flag=False, group_indicator=None, n_iter=30, random_state=42):
    """
    XGBoost hyperparameter tuning with multiple scoring metrics using RandomizedSearchCV.

    Returns:
        best_model: trained XGBClassifier
        best_params: dict of best hyperparameters
        results_df: DataFrame with scores for all tested hyperparameter combinations
    """

    kappa_scorer = make_scorer(cohen_kappa_score)
    specificity_scorer = make_scorer(recall_score, pos_label=0)

    scoring_metrics = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy',
        'F1': 'f1',
        'Sensitivity': 'recall_macro',
        'Precision': 'precision',
        'Specificity': specificity_scorer,
        'PRC': 'average_precision',
        'Kappa':  kappa_scorer,

    }

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Define distributions for random search
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.3),
        'scale_pos_weight': [1, pos_weight]
    }

    if split_by_group_flag:
        cv_strategy = StratifiedGroupKFold(n_splits=5)
    else:
        cv_strategy = StratifiedKFold(n_splits=5)

    xgb = XGBClassifier(eval_metric='logloss')

    # Here we preform the search itself
    # We decided to evaluate our model by the PRC.
    # PRC represents the potential of the model in regard to the positive (which is the minority) group.
    # By finding the point that maximizes the F1 score in the PRC column, we can reach to the pont that hold the best potential F1 score,
    # which may indicate the best balance between sensitivity and precision
    random_search = RandomizedSearchCV(
        xgb,
        param_distributions,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring_metrics,
        refit= 'average_precision',
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True
    )

    if split_by_group_flag:
        random_search.fit(X_train, y_train, groups=group_indicator)
    else:
        random_search.fit(X_train, y_train)

    # Convert results to DataFrame
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df[result_cols_to_save]

    return random_search.best_estimator_, random_search.best_params_, results_df

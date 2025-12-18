from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer, recall_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
import pandas as pd

cols_to_save = [
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
def logistic_grid_search_multi(X_train, y_train, cv=5):
    """
    Logistic Regression hyperparameter tuning with multiple scoring metrics using GridSearchCV.
    cv - amount of folds for cross validation

    Returns:
        best_model: trained LogisticRegression with best params
        best_params: dict of best hyperparameters
        results_df: DataFrame of all tested hyperparameter combinations and scores
    """
    kappa_scorer = make_scorer(cohen_kappa_score)
    specificity_scorer = make_scorer(recall_score, pos_label=0)

    scoring_metrics = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy',
        'F1': 'f1_macro',
        'Sensitivity': 'recall_macro',
        'Precision': 'precision',
        'specificity': specificity_scorer,
        'PRC': 'average_precision',
        'Kappa': kappa_scorer,

    }

    # Define valid hyperparameter grid
    param_grid = [
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100],'class_weight':[None, 'balanced']},
        {'solver': ['lbfgs', 'newton-cg'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100],'class_weight':[None, 'balanced']},
        {'solver': ['saga'], 'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9], 'class_weight':[None, 'balanced']},
    ]

    lr = LogisticRegression(max_iter=1000)

    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=cv,
        scoring=scoring_metrics,
        refit='AUC',  # choose metric to pick best model
        return_train_score=True,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[result_cols_to_save]

    return grid_search.best_estimator_, grid_search.best_params_, results_df

def logistic_random_search_multi(X_train, y_train, cv=5, n_iter=20, random_state=42):
    """
    Logistic Regression hyperparameter tuning with multiple scoring metrics using RandomizedSearchCV.
    n_iter - is the number of random hyperparameter combinations to try

    Returns:
        best_model: trained LogisticRegression with best params
        best_params: dict of best hyperparameters
        results_df: DataFrame of all tested hyperparameter combinations and scores
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

    # Define search space
    param_distributions = [
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': loguniform(1e-3, 1e3),'class_weight':[None, 'balanced']},
        {'solver': ['lbfgs', 'newton-cg'], 'penalty': ['l2'], 'C': loguniform(1e-3, 1e3),'class_weight':[None, 'balanced']},
        {'solver': ['saga'], 'penalty': ['elasticnet'], 'C': loguniform(1e-3, 1e3), 'l1_ratio': [0.1, 0.5, 0.9], 'class_weight':[None, 'balanced']},
    ]

    lr = LogisticRegression(max_iter=1000)

    random_search = RandomizedSearchCV(
        lr,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring_metrics,
        refit='AUC',
        return_train_score=True,
        n_jobs=-1,
        random_state=random_state
    )

    random_search.fit(X_train, y_train)

    # Convert results to DataFrame
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df[result_cols_to_save]

    return random_search.best_estimator_, random_search.best_params_, results_df

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
import pandas as pd

scoring_metrics = {
    'AUC': 'roc_auc',
    'Accuracy': 'accuracy',
    'F1': 'f1_macro',
    'Sensitivity': 'recall_macro'
}

def logistic_grid_search_multi(X_train, y_train, cv=5):
    """
    Logistic Regression hyperparameter tuning with multiple scoring metrics using GridSearchCV.
    cv - amount of folds for cross validation

    Returns:
        best_model: trained LogisticRegression with best params
        best_params: dict of best hyperparameters
        results_df: DataFrame of all tested hyperparameter combinations and scores
    """

    # Define valid hyperparameter grid
    param_grid = [
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]},
        {'solver': ['lbfgs', 'newton-cg'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100]},
        {'solver': ['saga'], 'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}
    ]

    lr = LogisticRegression(max_iter=1000)

    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=cv,
        scoring=scoring_metrics,
        refit='AUC',  # choose metric to pick best model
        return_train_score=False,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[[
        'params',
        'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_F1', 'mean_test_Sensitivity'
    ]]

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

    # Define search space
    param_distributions = [
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': loguniform(1e-3, 1e3)},
        {'solver': ['lbfgs', 'newton-cg'], 'penalty': ['l2'], 'C': loguniform(1e-3, 1e3)},
        {'solver': ['saga'], 'penalty': ['elasticnet'], 'C': loguniform(1e-3, 1e3), 'l1_ratio': [0.1, 0.5, 0.9]}
    ]

    lr = LogisticRegression(max_iter=1000)

    random_search = RandomizedSearchCV(
        lr,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring_metrics,
        refit='AUC',
        return_train_score=False,
        n_jobs=-1,
        random_state=random_state
    )

    random_search.fit(X_train, y_train)

    # Convert results to DataFrame
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df[[
        'params',
        'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_F1', 'mean_test_Sensitivity'
    ]]

    return random_search.best_estimator_, random_search.best_params_, results_df

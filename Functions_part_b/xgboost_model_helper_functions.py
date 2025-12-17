from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from xgboost import XGBClassifier
import pandas as pd

scoring_metrics = {
    'AUC': 'roc_auc',
    'Accuracy': 'accuracy',
    'F1': 'f1_macro',
    'Sensitivity': 'recall'
}

def xgb_grid_search_multi(X_train, y_train, cv=5):
    """
    XGBoost hyperparameter tuning with multiple scoring metrics using GridSearchCV.

    Returns:
        best_model: trained XGBClassifier
        best_params: dict of best hyperparameters
        results_df: DataFrame with scores for all tested hyperparameter combinations
    """

    # Define a practical hyperparameter grid for XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=cv,
        scoring=scoring_metrics,
        refit='AUC',
        n_jobs=-1,
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)

    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[[
        'params',
        'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_F1', 'mean_test_Sensitivity'
    ]]

    return grid_search.best_estimator_, grid_search.best_params_, results_df

def xgb_random_search_multi(X_train, y_train, cv=5, n_iter=30, random_state=42):
    """
    XGBoost hyperparameter tuning with multiple scoring metrics using RandomizedSearchCV.

    Returns:
        best_model: trained XGBClassifier
        best_params: dict of best hyperparameters
        results_df: DataFrame with scores for all tested hyperparameter combinations
    """

    # Define distributions for random search
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.3)
    }

    xgb = XGBClassifier(eval_metric='logloss')

    random_search = RandomizedSearchCV(
        xgb,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring_metrics,
        refit='AUC',
        n_jobs=-1,
        random_state=random_state,
        return_train_score=False
    )

    random_search.fit(X_train, y_train)

    # Convert results to DataFrame
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df[[
        'params',
        'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_F1', 'mean_test_Sensitivity'
    ]]

    return random_search.best_estimator_, random_search.best_params_, results_df

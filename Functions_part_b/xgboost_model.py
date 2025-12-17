from xgboost import XGBClassifier

from .xgboost_model_helper_functions import xgb_grid_search_multi, xgb_random_search_multi

def find_best_hp_xgboost(X_train, y_train, split_name):

    # Grid search
    # best_xgb_grid, best_params_grid, results_grid = xgb_grid_search_multi(X_train, y_train)
    # results_grid.to_excel(f'{split_name}_xgboost_results_xgb_grid.xlsx')
    # print(f'Saved {split_name}_xgboost_results_xgb_grid.xlsx')

    # Randomized search
    best_xgb_rand, best_params_rand, results_rand = xgb_random_search_multi(X_train, y_train)
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
    return model
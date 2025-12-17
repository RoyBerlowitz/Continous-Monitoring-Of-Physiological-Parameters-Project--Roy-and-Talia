from .xgboost_model_helper_functions import xgb_grid_search_multi, xgb_random_search_multi

def train_xgboost(X_train, y_train):

    # Grid search
    # best_xgb_grid, best_params_grid, results_grid = xgb_grid_search_multi(X_train, y_train)
    # print("XGBoost Grid Search Best Params:", best_params_grid)
    # print(results_grid.head())
    # results_grid.to_excel('results_xgb_grid.xlsx')

    # Randomized search
    best_xgb_rand, best_params_rand, results_rand = xgb_random_search_multi(X_train, y_train)
    print("XGBoost Randomized Search Best Params:", best_params_rand)
    print(results_rand.head())
    results_rand.to_excel('results_xgb_rand.xlsx')

    return best_xgb_rand
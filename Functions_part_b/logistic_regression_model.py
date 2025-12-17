from sklearn.linear_model import LogisticRegression

from .logistic_regression_model_helper_functions import logistic_random_search_multi, logistic_grid_search_multi

def find_best_hp_logistic_regression(X_train, y_train, split_name):

    # # Grid search
    # best_model_grid, best_params_grid, results_grid = logistic_grid_search_multi(X_train, y_train)
    # results_grid.to_excel(f'{split_name}_logistic_results_grid_search.xlsx', index=False)
    # print(f'Saved {split_name}_results_grid_search.xlsx')

    # Randomized search
    best_model_rand, best_params_rand, results_rand = logistic_random_search_multi(X_train, y_train)
    results_rand.to_excel(f'{split_name}_logistic_results_rand_search.xlsx', index=False)
    print(f'Saved {split_name}_logistic_results_rand_search.xlsx')

    return best_params_rand

def train_logistic_regression(X_train, y_train, best_hp):

    model = LogisticRegression(
        max_iter=1000,
        **best_hp
    )

    model.fit(X_train, y_train)
    return model

#possible hyperparams
#Regularization
    #penalty: Type of regularization
        # 'l2' (default) → Ridge (most common)
        # 'l1' → Lasso (feature selection)
        # 'elasticnet' → mix of L1 and L2
        # 'none' → no regularization
    #C: inverse of regularization strength:
        #Smaller C → stronger regularization
        #Larger C → weaker regularization
        #usually try C=[0.001, 0.01, 0.1, 1, 10, 100]
    #l1_ratio (only if penalty='elasticnet')
        # l1_ratio = 0 → pure L2
        # l1_ratio = 1 → pure L1
#Optimization / Solver
    #solver: Optimization algorithm
        # 'lbfgs'(default, fast, L2 only)
        # 'liblinear'(small datasets, L1 or L2)
        # 'saga'(large datasets, supports L1 / L2 / elasticnet)
        # 'newton-cg'(L2 only)
    #max_iter:Maximum number of iterations
        # 100  # default. Increase if you see convergence warnings.
    #tol: Tolerance for stopping criteria
        #1e-4. Smaller → more precise but slower.
#Class Handling (important for imbalanced data)
    # class_weight
        # None
        # 'balanced' → adjusts weights inversely proportional to class frequency
        # or a dict {0: w0, 1: w1}
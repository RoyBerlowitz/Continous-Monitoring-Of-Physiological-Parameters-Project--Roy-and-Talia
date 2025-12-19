from sklearn.linear_model import LogisticRegression

from .logistic_regression_model_helper_functions import logistic_random_search_multi, logistic_grid_search_multi
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import numpy as np
from .evaluate_model_functions import closest_point_roc

def find_best_hp_logistic_regression(X_train, y_train, split_name, split_by_group_flag = False, group_indicator = None):

    # # Grid search
    # best_model_grid, best_params_grid, results_grid = logistic_grid_search_multi(X_train, y_train)
    # results_grid.to_excel(f'{split_name}_logistic_results_grid_search.xlsx', index=False)
    # print(f'Saved {split_name}_results_grid_search.xlsx')

    # Randomized search
    best_model_rand, best_params_rand, results_rand = logistic_random_search_multi(X_train, y_train, split_by_group_flag=split_by_group_flag, group_indicator=group_indicator)
    results_rand.to_excel(f'{split_name}_logistic_results_rand_search.xlsx', index=False)
    print(f'Saved {split_name}_logistic_results_rand_search.xlsx')

    return best_params_rand

def train_logistic_regression(X_train, y_train, best_hp):

    model = LogisticRegression(
        max_iter=1000,
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
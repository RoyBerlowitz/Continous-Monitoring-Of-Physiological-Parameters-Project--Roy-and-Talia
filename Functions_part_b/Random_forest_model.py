import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold,StratifiedKFold
from sklearn.metrics import roc_curve, cohen_kappa_score, make_scorer, recall_score, precision_recall_curve, average_precision_score
import numpy as np
from .evaluate_model_functions import closest_point_roc



def find_best_random_forrest_parameters (train_df, train_labels, group_indicator, n_jobs = -1, n_iterations = 50, split_name = "Individual Split", split_by_group_flag = False, wrapper_text = ''):
    # Here we preform the search for the best hyperparameters for the SVM model.
    # We will preform parallel run to accelerate time

    # we start by adjusting the dimension of the validation labels.
    train_target = train_labels.values.ravel()
    # We create a pipeline of how we process the data in the stages of training the model
    pipeline = Pipeline([('Random_Forest', RandomForestClassifier(random_state=42))])

    # We will determine the ranges of the values of the parameters.
    # we preform Random Search instead of GridSearch, as it allows to cover the space more completely.
    # we look for the parameters that will award us with the best combination between good prediction and no overfitting.
    # all the parameters are randomly chosen as integers from a declined range

    params_ranges = {'Random_Forest__n_estimators': randint(100, 500), #how many trees in the ensemble give us the best result
                     'Random_Forest__max_depth': randint(10, 41), # we look for the best depth, and limit the depth to prevent overfitting
                     'Random_Forest__min_samples_split': randint(20, 100), #the minimal amount of sample to split - to prevent overfitting while catching the data patterns
                     'Random_Forest__max_samples': uniform(0.5, 0.4), #max sample determine which percent of the data will be transmitted to each tree
                     'Random_Forest__min_samples_leaf': randint(10, 100), #the minimal amount of sample in final leave - to prevent overfitting while catching the data patterns
                     'Random_Forest__class_weight': ['balanced', 'balanced_subsample']     # class weight is examined to be between balanced_subsample and balanced which to under the level of balance we should create in the model
                    }

    # we add scoring metrics we will examine in the Excel.
    # we chose AUC, Accuracy, F1_score, PRC, specificity, precision, and sensitivity
    # We added also cohen's kappa as it is more informative regarding the bias of the model towards the majority group
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

    # Here We determine the stratified K-Folds strategy.
    # For the group split, we will use a strategy that ensure the division is made in a way that 20% of the groups are the test in each iteration
    if split_by_group_flag:
        cv_strategy = StratifiedGroupKFold(n_splits=5, random_state=42)
    else:
        cv_strategy = StratifiedKFold(n_splits=5, random_state=42)

    # Here we preform the search itself
    # We decided to evaluate our model by the PRC.
    # PRC represents the potential of the model in regard to the positive (which is the minority) group.
    # By finding the point that maximizes the F1 score in the PRC column, we can reach to the pont that hold the best potential F1 score,
    # which may indicate the best balance between sensitivity and precision

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=params_ranges,  # the parameters we look for
        n_iter=n_iterations,  # number of iterations to check
        cv=cv_strategy,  # stratified K-folds to look for
        scoring=scoring_metrics,  # we find all the wanted metrics
        refit='PRC',  # AUC-ROC is the evaluation metric
        n_jobs=n_jobs,
        verbose=3,
        random_state=42,  # to have consistent results
        return_train_score=True

    )

    print(f"Starting Randomized Search with {n_iterations} iterations and 5-Fold Cross-Validation...")
    # we fit each option with the data. if the division is by groups, we indicate it what group to look for
    if split_by_group_flag:
        random_search.fit(train_df, train_target, groups=group_indicator)
    else:
        random_search.fit(train_df, train_target)

    # we find the best parameters
    best_parameters = random_search.best_params_
    # we export to Excel the metric score for each parameter for later intrepretabillity
    # we save both test and train values to be able to identify overfitting
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    print("Available columns in results:", cv_results_df.columns.tolist())
    cols_to_save = [
        'params',
        # TRAIN SCORE
        'mean_train_AUC', 'mean_train_Accuracy', 'mean_train_Specificity', 'mean_train_Sensitivity',
        'mean_train_Precision', 'mean_train_F1', 'mean_train_PRC', 'mean_train_Kappa',
        # TEST SCORES
        'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_Specificity', 'mean_test_Precision',
        'mean_test_Sensitivity', 'mean_test_F1', 'mean_test_PRC', 'mean_test_Kappa',
        # Control columns
        'mean_fit_time', 'rank_test_PRC'
    ]
    cv_results_filtered = cv_results_df[cols_to_save].sort_values(by='rank_test_PRC')

    excel_file_name = split_name + wrapper_text + 'Random_Forrest_Search_Results.xlsx'
    cv_results_filtered.to_excel(excel_file_name, index=False)
    print(f"\n--- CV Results Saved to: {excel_file_name} ---")
    # we return the best model
    return best_parameters

def train_random_forest_classifier (train_df, train_labels, best_parameters, name = "Individual Split", n_jobs = -1):
    # This function is meant to fit the model with the selected hyperparameters to the data
    # we start by adjusting the dimension of the validation labels.

    train_target = train_labels.values.ravel()
    # we get the best params
    max_depth = best_parameters['Random_Forest__max_depth']
    n_estimators = best_parameters['Random_Forest__n_estimators']
    min_samples_split = best_parameters['Random_Forest__min_samples_split']
    class_weight = best_parameters['Random_Forest__class_weight']
    max_samples = best_parameters['Random_Forest__max_samples']
    min_sample_leaf = best_parameters.get('Random_Forest__min_samples_leaf')
    #we create the pipeline again
    Random_Forest_pipeline = Pipeline([("Random_Forest", RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf= min_sample_leaf, class_weight = class_weight, max_samples = max_samples, random_state=42, oob_score=True, n_jobs =n_jobs))])
    print (f"Starting Training the Random Forrest for {name}...")
    # we fit the model to the data
    best_Random_Forest_pipeline = Random_Forest_pipeline.fit(train_df, train_target)
    best_Random_Forest_Model = best_Random_Forest_pipeline.named_steps['Random_Forest']

    # Here, we try to use the power of the PRC curve to find the best operating point in regard of F1.
    # To use the PRC without overfitting which is based on the fact we find the best operating point with the data we trained on,
    # we use the OOB - each tree is not fed with the entire data, so the OOB data is the data the tree was not trained on.
    # we use this data to predict the probabilities and by that find the best operating point.
    oob_probs = best_Random_Forest_Model.oob_decision_function_[:, 1]
    # we calculate the needed calculation for the PRC curve
    precisions, recalls, thresholds = precision_recall_curve(train_target, oob_probs)
    avg_prec = average_precision_score(train_target, oob_probs)

    # Here we find the optimal threshold, which is the point which gives the best F1 score.
    # F1 score represnt both senstivity and precision and by that hints a lot about the minority group
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_Random_Forest_Model.optimal_threshold_PRC_ = thresholds[best_idx]

    # We will also find the ROC-AUC optimal point - which is the closet one to the (0,1)
    fpr, tpr, roc_thresholds = roc_curve(train_target, oob_probs)
    roc_res = closest_point_roc(fpr, tpr, roc_thresholds)
    best_Random_Forest_Model.optimal_threshold_ROC_ = roc_res['threshold']

    return best_Random_Forest_Model


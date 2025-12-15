import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV


def find_best_random_forrest_parameters (train_df, train_labels,n_jobs = -1, n_iterations = 50, split_name = "Individual Split"):
    train_target = train_labels.values.ravel()

    pipeline = Pipeline(['Random Forest Classifier', RandomForestClassifier(random_state=42)])
    params_ranges = {'Random_Forest__n_estimators': randint(100, 500),
                     'Random_Forest__max_depth': [None]  + randint(10, 50),
                     'Random_Forest__min_samples_per_split': randint(2, 20),
    }
    scoring_metrics = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy',
        'F1': 'f1_macro',
        'Sensitivity': 'recall_macro'
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=params_ranges,  # the parameters we look for
        n_iter=n_iterations,  # number of iterations to check
        cv=5,  # 5 stratified K-folds to look for
        scoring=scoring_metrics, # we find all the wanted metrics
        refit='AUC',  # AUC-ROC is the evaluation metric
        n_jobs=n_jobs,
        verbose=1,
        random_state=42  # to have consistent results
    )

    print(f"Starting Randomized Search with {n_iterations} iterations and 5-Fold Cross-Validation...")
    random_search.fit(train_df, train_target)

    best_parameters = random_search.best_params_

    cv_results_df = pd.DataFrame(random_search.cv_results_)

    cols_to_save = ['params', 'mean_train_AUC', 'mean_train_Accuracy', 'mean_train_Sensitivity', 'mean_train_F1',
                    'mean_fit_time']

    cv_results_filtered = cv_results_df[cols_to_save].sort_values(by='rank_test_AUC')

    excel_file_name = split_name + 'Random_Forrest_Search_Results.xlsx'
    cv_results_filtered.to_excel(excel_file_name, index=False)
    print(f"\n--- CV Results Saved to: {excel_file_name} ---")
    # we return the best model
    return best_parameters

def train_random_forest_classifer (train_df, train_labels, best_parameters, name = "Individual Split"):
    train_target = train_labels.values.ravel()

    max_depth = best_parameters['Random_Forest__max_depth']
    n_estimators = best_parameters['Random_Forest__n_estimators']
    min_samples_split = best_parameters['Random_Forest__min_samples_per_split']

    Random_Forest_pipeline = Pipeline(["Random Forest Classifier", RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, random_state=42)])

    print (f"Starting Training the Random Forrest for {name}...")
    best_Random_Forest_Model = Random_Forest_pipeline.fit(train_df, train_target)

    return best_Random_Forest_Model


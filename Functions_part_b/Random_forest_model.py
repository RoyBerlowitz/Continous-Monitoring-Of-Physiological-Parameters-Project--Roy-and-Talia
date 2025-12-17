import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer


def find_best_random_forrest_parameters (train_df, train_labels,n_jobs = -1, n_iterations = 50, split_name = "Individual Split"):
    # Here we preform the search for the best hyperparameters for the SVM model.
    # We will preform parallel run to accelerate time

    # we start by adjusting the dimension of the validation labels.
    train_target = train_labels.values.ravel()
    # We create a pipeline of how we process the data in the stages of training the model
    pipeline = Pipeline(['Random Forest Classifier', RandomForestClassifier(random_state=42)])

    # We will determine the ranges of the values of the parameters.
    # we preform RandomSearch instead of GridSearch, as it allows to cover the space more completely.
    # we look for the parameters that will award us with the best combination between good prediction and no overfitting.
    # all of the are randomly chosen as integers  from a declined range
    # class weight is examined to be between None and balanced which gives more weight the minority group and by that creates more balanced model
    params_ranges = {'Random_Forest__n_estimators': randint(100, 500),
                     'Random_Forest__max_depth': [None]  + randint(10, 50),
                     'Random_Forest__min_samples_per_split': randint(2, 20),
                     'Random_Forest__class_weight': [None, 'balanced'] }

    # we add scoring metrics we will examine in the Excel.
    # we chose AUC, Accuracy, F1_score and sensitivity
    # We added also cohen's kappa as it is more informative regarding the bias of the model towards the majority group
    kappa_scorer = make_scorer(cohen_kappa_score)
    scoring_metrics = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy',
        'F1': 'f1_macro',
        'Sensitivity': 'recall_macro',
        'PRC': 'average_precision',
        'Kappa':  kappa_scorer
    }

    # Here we preform the search itself
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=params_ranges,  # the parameters we look for
        n_iter=n_iterations,  # number of iterations to check
        cv=5,  # 5 stratified K-folds to look for
        scoring=scoring_metrics, # we find all the wanted metrics
        refit='AUC',  # AUC-ROC is the evaluation metric
        n_jobs=n_jobs,
        verbose=3,
        random_state=42,  # to have consistent results
        return_train_score=True

    )

    print(f"Starting Randomized Search with {n_iterations} iterations and 5-Fold Cross-Validation...")
    #we fit each option with the data
    random_search.fit(train_df, train_target)
    # we find the best parameters
    best_parameters = random_search.best_params_
    # we export to Excel the metric score for each parameter for later intrepretabillity
    # we save both test and train values to be able to identify overfitting
    cv_results_df = pd.DataFrame(random_search.cv_results_)

    cols_to_save = ['params',

    # TRAIN SCORE
    'mean_train_AUC', 'mean_train_Accuracy', 'mean_train_Sensitivity', 'mean_train_F1',
    'mean_train_PRC', 'mean_train_Kappa',

    # TEST SCORES
    'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_Sensitivity', 'mean_test_F1',
    'mean_test_PRC', 'mean_test_Kappa',

    # Control columns
    'mean_fit_time',
    'rank_test_AUC']

    cv_results_filtered = cv_results_df[cols_to_save].sort_values(by='rank_test_AUC')

    excel_file_name = split_name + 'Random_Forrest_Search_Results.xlsx'
    cv_results_filtered.to_excel(excel_file_name, index=False)
    print(f"\n--- CV Results Saved to: {excel_file_name} ---")
    # we return the best model
    return best_parameters

def train_random_forest_classifier (train_df, train_labels, best_parameters, name = "Individual Split"):
    # This function is meant to fit the model with the selected hyperparameters to the data
    # we start by adjusting the dimension of the validation labels.

    train_target = train_labels.values.ravel()
    # we get the best params
    max_depth = best_parameters['Random_Forest__max_depth']
    n_estimators = best_parameters['Random_Forest__n_estimators']
    min_samples_split = best_parameters['Random_Forest__min_samples_per_split']
    class_weight = best_parameters['Random_Forest__class_weight']
    #we create the pipeline again
    Random_Forest_pipeline = Pipeline(["Random Forest Classifier", RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, class_weight = class_weight, random_state=42)])

    print (f"Starting Training the Random Forrest for {name}...")
    # we fit the model to the data
    best_Random_Forest_Model = Random_Forest_pipeline.fit(train_df, train_target)

    return best_Random_Forest_Model


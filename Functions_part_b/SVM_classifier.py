import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold,StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy.stats import loguniform, randint
from sklearn.metrics import roc_curve, cohen_kappa_score, make_scorer, recall_score, precision_recall_curve, average_precision_score
from .evaluate_model_functions import closest_point_roc


def perform_PCA(train_df, target, n_dimensions, name = "Individual Split"):
    # At first, we preform a PCA to reduce dimensionality.
    # This can be helpful for visualization as well as reducing dimensionality for SVM
    # the n_dimensions will define the number of principles that remains
    pca = PCA(n_components=n_dimensions)

    # We preform the PCA transformation on the data
    principal_components = pca.fit_transform(train_df)
    columns = [f'Principal Component {i + 1}' for i in range(n_dimensions)]
    pca_df = pd.DataFrame(data=principal_components, columns=columns)
    pca_df['labels'] = target
    # we plot the PCA component results
    sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue='labels',
        palette='Set1',
        data=pca_df,
        s=100,
        alpha=0.7
    )
    plt.title(f'PCA of {name}', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='labels', fontsize=10)
    plt.grid()
    plt.show()

    return pca_df


def find_best_SVM_parameters(train_df, train_labels, group_indicator, n_jobs = -1, n_iterations = 50, split_name = "Individual Split", split_by_group_flag = False):

    # Here we preform the search for the best hyperparameters for the SVM model.
    # We will preform parallel run to accelerate time

    # we start by adjusting the dimension of the validation labels.
    train_target = train_labels.values.ravel()
    # We create a pipeline of how we process the data in the stages of training the model
    if len(train_df.columns) >=5: #to deal with different number of selected features
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # we normalize again to prevent data leakage and ensure normal data distribution
            ('pca', PCA(n_components=0.95)),  # we preform PCA to reduce dimensionality
            ('svm', SVC(kernel='rbf', random_state=42, probability=True))  # we preform SVM with RBF kernel
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # we normalize again to prevent data leakage and ensure normal data distribution
            ('svm', SVC(kernel='rbf', random_state=42, probability=True))  # we preform SVM with RBF kernel
        ])

    # We will determine the ranges of the values of the parameters.
    # we preform RandomSearch instead of GridSearch, as it allows to cover the space more completely.
    # for Gamma and C, we will search using uniform distribuation with logarithmic scale
    # for the n_components, we will try different values
    # class weight is examined to be between None and balanced which gives more weight the minority group and by that creates more balanced model
    if len(train_df.columns) >=5: #to deal with different number of selected features
        params_ranges = {"svm__C": loguniform(0.01, 30),
                         'svm__gamma': loguniform(0.0001, 0.5),
                         'pca__n_components': randint(3, len(train_df.columns) - 1),
                         'svm__class_weight': ['balanced'] }
    else:
        params_ranges = {"svm__C": loguniform(0.01, 30),
                         'svm__gamma': loguniform(0.0001, 0.5),
                         'svm__class_weight': ['balanced'] }


    # we add scoring metrics we will examine in the Excel.
    # we chose AUC, Accuracy, F1_score and sensitivity
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
        cv_strategy = StratifiedGroupKFold(n_splits=3)
    else:
        cv_strategy = StratifiedKFold(n_splits=3)

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

    print(f"Starting Randomized Search with {n_iterations} iterations and 3-Fold Cross-Validation...")
    # we fit each option with the data. if the division is by groups, we indicate it what group to look for
    if split_by_group_flag:
        random_search.fit(train_df, train_target, groups=group_indicator)
    else:
        random_search.fit(train_df, train_target)


    # metric evaluation
    print("\n--- Results ---")
    print("Best parameters found:")
    print(random_search.best_params_)

    # We save the best parameters according to the evaluation metric
    best_parameters = random_search.best_params_
    #best_model = random_search.best_estimator_
    # we export to Excel the metric score for each parameter
    cv_results_df = pd.DataFrame(random_search.cv_results_)

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

    excel_file_name = split_name +'_SVM_Random_Search_Results.xlsx'
    cv_results_filtered.to_excel(excel_file_name, index=False)
    print(f"\n--- CV Results Saved to: {excel_file_name} ---")
    #we return the best model
    return best_parameters

def train_SVM(train_df, train_labels, best_parameters, name = "Individual Split", split_by_group_flag = False):
    #Here, we preform the SVM on the validation set, according to the SVM we found.
    # we start by adjusting the dimension of the validation labels.
    train_target = train_labels.values.ravel()
    # we normalize again to prevent data leakage and ensure normal data distribution
    steps = [
        ('scaler', StandardScaler())
    ]
    # we seperated to cases in which we want to preform PCA and not. so we check if PCA was done.
    # if done, we preform PCA to reduce dimensionality, with the found number of components
    if 'pca__n_components' in best_parameters:
        pca_n_components = best_parameters['pca__n_components']
        steps.append(('pca', PCA(n_components=pca_n_components)))

    #we insert the parameters of the best model
    gamma = best_parameters['svm__gamma']
    C = best_parameters['svm__C']
    #pca_n_components = best_parameters['pca__n_components']
    class_weights = best_parameters['svm__class_weight']
    # we preform SVM with RBF kernel and the parameters we found
    steps.append(('SVM', SVC(
        kernel='rbf',
        C=C,
        gamma=gamma,
        random_state=42,
        probability=True,
        class_weight=class_weights
    )))

    best_SVM_parameters = Pipeline(steps)

    print (f"Starting Training the SVM model for {name}...")
    # We fit the model with our parameters to the data
    best_SVM_pipeline = best_SVM_parameters.fit(train_df, train_target)
    best_SVM_model = best_SVM_pipeline.named_steps['SVM']

    # Here, we try to use the power of the PRC curve to find the best operating point in regard of F1.
    # we face a challenge - we try to estimate the PRC without overfitting, which is non-trivial based on the fact we find the best operating point with the data we trained on.
    # we use the Cross-validation prediction - we train again but in a 5-folds scheme, so we get each time the probabilities on data the model "did not see".
    # we use the result to predict the probabilities and by that find the best operating point.
    # it is not exactly the same model, but it is close and justified estimation.
    # we preserve the same logic regarding the group k-folds also here
    if split_by_group_flag:
        cv_strategy = StratifiedGroupKFold(n_splits=5)
    else:
        cv_strategy = StratifiedKFold(n_splits=5)
    y_probs = cross_val_predict(best_SVM_model, train_df, train_target, cv=cv_strategy, method='predict_proba')[:, 1]
    # we calculate the needed calculation for the PRC curve
    precisions, recalls, thresholds = precision_recall_curve(train_target, y_probs)
    avg_prec = average_precision_score(train_target, y_probs)

    # Here we find the optimal threshold, which is the point which gives the best F1 score.
    # F1 score represnt both senstivity and precision and by that hints a lot about the minority group
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_SVM_model.optimal_threshold_PRC_ = thresholds[best_idx]

    # We will also find the ROC-AUC optimal point - which is the closet one to the (0,1)
    fpr, tpr, roc_thresholds = roc_curve(train_target, y_probs)
    roc_res = closest_point_roc(fpr, tpr, roc_thresholds)
    best_SVM_model.optimal_threshold_ROC_ = roc_res['threshold']

    return best_SVM_model

    # # We find two predictions. one is the predicted label, the second one is the probability to be in the 1 label for each sample.
    # predicted_validation = best_SVM_model.predict(val_df)
    # val_scores = best_SVM_model.predict_proba(val_df)[:, 1]
    #
    # # We compute the AUC for the model
    # val_auc = roc_auc_score(val_target, val_scores)
    #
    #
    # print(f"\nTest Set AUC-ROC Score (Best Model) for {name}: {val_auc:.4f}")
    # print("\nClassification Report:")
    # # we show the classification report that includes various metrices
    # print(classification_report(val_target, predicted_validation))



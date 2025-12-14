import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.metrics import roc_auc_score, classification_report, recall_score, accuracy_score
from scipy.stats import randint


def preform_PCA(train_df, target, n_dimensions, name = "Individual Split"):
    # At first, we preform a PCA to reduce dimensionality.
    # This can be helpful for visualization as well as reducing dimensionality for SVM
    # the n_dimensions will define the number of principles that remains
    pca = PCA(n_components=n_dimensions)
    #
    # We preform the PCA transformation on the data
    principal_components = pca.fit_transform(train_df)
    pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['labels'] = target

    sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue='Target',
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

#
def SVM_hyperparametrs_fit(train_df, train_labels, n_jobs = -1, n_iterations = 50, split_name = "Individual Split"):
    # Here we preform the search for the best parameters for the SVM model.
    # We will preform parallel run to accelerate time

    # we start by adjusting the dimension of the validation labels.
    train_target = train_labels.values.ravel()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # we normalize again to prevent data leakage and ensure normal data distribution
        ('pca', PCA(n_components=0.95)),  # we preform PCA to reduce dimensionality
        ('svm', SVC(kernel='rbf', random_state=42))  # we preform SVM with RBF kernel
    ])

    # We will determine the ranges of the values of the parameters.
    # we preform RandomSearch instead of GridSearch, as it allows to cover the space more completely.
    # for Gamma and C, we will search using uniform distribuation with logarithmic scale
    # for the n_components, we will try different values
    params_ranges = {"C_range": loguniform(0.01, 100),'gamma_range': loguniform(0.0001, 10), 'pca__n_components': randint(5, len(train_df.columns) - 1)}

    # we add scoring metrics we will examine in the Excel.
    # we chose AUC, Accuracy, F1_score and sensitivity
    scoring_metrics = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy',
        'F1': 'f1_macro',
        'Sensitivity': 'recall_macro'
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=params_ranges,  # 🛑 שימוש ב-distributions
        n_iter=n_iterations,  # number of iterations to check
        cv=5,  # 5 stratified K-folds to look for
        scoring=scoring_metrics, # we find all the wanted metrics
        refit='AUC',  # AUC-ROC is the evaluation metric
        n_jobs=n_jobs,
        verbose=-1,
        random_state=42  # to have consistent results
    )

    # we preform the SVM hyper-parameters search
    print(f"Starting Randomized Search with {n_iterations} iterations and 5-Fold Cross-Validation...")
    random_search.fit(train_df, train_target)

    # metric evaluation
    print("\n--- Results ---")
    print("Best parameters found:")
    print(random_search.best_params_)

    # We define the best model according to the evaluation metric
    best_model = random_search.best_estimator_
    # we export to Excel the metric score for each parameter
    cv_results_df = pd.DataFrame(random_search.cv_results_)

    cols_to_save = [
        'params',
        'mean_train_AUC', 'mean_train_Accuracy', 'mean_train_Sensitivity',  'mean_train_F1'
        'mean_fit_time'
    ]

    cv_results_filtered = cv_results_df[cols_to_save].sort_values(by='rank_test_AUC')

    excel_file_name = split_name +'_SVM_Random_Search_Results.xlsx'
    cv_results_filtered.to_excel(excel_file_name, index=False)
    print(f"\n--- CV Results Saved to: {excel_file_name} ---")
    #we return the best model
    return best_model

def preform_SVM( val_df, val_labels, best_SVM_model):
    #Here, we preform the SVM on the validation set, according to the SVM we found.
    # we start by adjusting the dimenstion of the validation labels.
    val_target = val_labels.values.ravel()

    # We find two predictions. one is the predicted label, the second one is the probability to be in the 1 label for each sample.
    predicted_validation = best_SVM_model.predict(val_df)
    val_scores = best_SVM_model.predict_proba(val_df)[:, 1]

    # We compute the AUC for the model
    val_auc = roc_auc_score(val_target, val_scores)



    print(f"\nTest Set AUC-ROC Score (Best Model): {val_auc:.4f}")
    print("\nClassification Report:")
    # we show the classification report that includes various metrices
    print(classification_report(val_scores, predicted_validation))



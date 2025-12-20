import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, cohen_kappa_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold, cross_val_predict


from .consts import ModelNames


def select_features_wrapper(train_df, train_labels, frozen_params,
                                        n_features_range=[3, 5, 7, 10, 12, 15],
                                        model_type=ModelNames.RANDOM_FOREST, split_name="Individual", split_by_group_flag = False):
    #We implemented wrapper selection based on the two best operation model: gradient boosting and random forrest.
    # we decided to look for the best number of features, and subsequecntly, the best combination for each of this number.
    # because preforming the search for the best hyperparameters for each number pf features will take an incerible amount of runtime,
    # we decided to look for the best Hyperparametrers computed for each split in the filter method, and preform the search on them.
    # although not ideal, it will give us the a sense of idea about how many features to take

    # first we adjust the labels dimensions to the correct one
    train_target = train_labels.values.ravel()
    results = []

    # Here we define the scores the model is considering while making the estimation.
    # Those are the metrics it will calculate.
    kappa_scorer = make_scorer(cohen_kappa_score)
    specificity_scorer = make_scorer(recall_score, pos_label=0)

    scoring = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy',
        'F1': 'f1',
        'Sensitivity': 'recall_macro',
        'Precision': 'precision',
        'Specificity': specificity_scorer,
        'PRC': 'average_precision',
        'Kappa': kappa_scorer,
    }

    # Now, we turn to create the models with the "frozen hyperparamters", the pre-selected hyperparameters
    # As wrapper method is established around specific type of model, we define the estimator differently accoriding to the model we examine
    if model_type == ModelNames.RANDOM_FOREST:
        # We clean the parametrs' names if thet are with Pipeline prefix
        clean_params = {k.split('__')[-1]: v for k, v in frozen_params.items()}
        # we define the estimator to be the random forest model with the selected parameters
        estimator = RandomForestClassifier(**clean_params, random_state=42, n_jobs=-1)
    elif model_type == ModelNames.XGBOOST:
        # We clean the parametrs' names if thet are with Pipeline prefix
        clean_params = {k.split('__')[-1]: v for k, v in frozen_params.items()}
        # we define the estimator to be the XG_boost model with the selected parameters
        estimator =  XGBClassifier(**clean_params, random_state=42, n_jobs=-1)

    print(f"Starting Wrapper Comparison for {split_name} with {model_type}...")

    for n in n_features_range:
        print(f"Checking performance with top {n} features...")

        # We use RFE for the features selection regarding the wanted number of features.
        # RFE removes the most redundant features in each iteration after creating the forest with all the features, until it reaches the most impactful feature
        selector = RFE(estimator=estimator, n_features_to_select=n, step=1, verbose=0)
        selector.fit(train_df, train_target)
        # we save the columns to a list
        selected_cols = train_df.columns[selector.support_].tolist()

        # we run cross validation the model with the "surviving" features
        # For the group split, we will use a strategy that ensure the division is made in a way that 20% of the groups are the test in each iteration
        if split_by_group_flag:
            cv_strategy = StratifiedGroupKFold(n_splits=5)
        else:
            cv_strategy = StratifiedKFold(n_splits=5)
        # we preform the cross validation with the estimator and the chosen features
        cv_results = cross_validate(
            estimator,
            train_df[selected_cols],
            train_target,
            cv=cv_strategy,
            scoring=scoring,
            return_train_score=True,  # '
            n_jobs=-1,
            verbose= 3
        )

        #  we get the averages for the specific n features
        row = {
            'n_features': n,
            'Selected_Features': ", ".join(selected_cols)
        }

        #  we add the metrics for the data we save
        for metric in scoring.keys():
            row[f'mean_train_{metric}'] = cv_results[f'train_{metric}'].mean()
            row[f'mean_test_{metric}'] = cv_results[f'test_{metric}'].mean()

        results.append(row)

    #  we create the excel
    results_df = pd.DataFrame(results)

    # we organize the column in a way that the excel will be best
    cols = ['n_features', 'Selected_Features'] + \
           [c for c in results_df.columns if 'mean_test' in c] + \
           [c for c in results_df.columns if 'mean_train' in c]
    # we save it to the excel
    file_name = f"{split_name}_{model_type}_Feature_Count_Comparison.xlsx"
    results_df[cols].to_excel(file_name, index=False)

    print(f"\n--- Results saved to: {file_name} ---")
    # we save the best row as the best PRC - this is the metric we evaluate the model by
    best_row = results_df.loc[results_df['mean_test_PRC'].idxmax()]
    return best_row['Selected_Features'].split(", ")


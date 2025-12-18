import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, cohen_kappa_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def wrapper_selection(train_df, train_labels, frozen_params,
                                        n_features_range=[3, 5, 7, 10, 12, 15],
                                        model_type='RF', split_name="Individual"):
    """
    שלב א: השוואה בין כמויות פיצ'רים שונות.
    שומר גם Train וגם Test עבור כל המדדים שביקשת.
    """
    train_target = train_labels.values.ravel()
    results = []

    # 1. הגדרת ה-Scorers כפי שביקשת
    kappa_scorer = make_scorer(cohen_kappa_score)
    specificity_scorer = make_scorer(recall_score, pos_label=0)

    scoring = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy',
        'F1': 'f1_macro',
        'Sensitivity': 'recall_macro',
        'Precision': 'precision',
        'Specificity': specificity_scorer,
        'PRC': 'average_precision',
        'Kappa': kappa_scorer,
    }

    # 2. בניית המודל עם הפרמטרים ה"קפואים"
    if model_type == 'RF':
        # ניקוי השמות של הפרמטרים במידה והם מגיעים עם תחילית של Pipeline
        clean_params = {k.split('__')[-1]: v for k, v in frozen_params.items()}
        estimator = RandomForestClassifier(**clean_params, random_state=42, n_jobs=-1)

    print(f"Starting Stage 1 Comparison for {split_name}...")

    for n in n_features_range:
        print(f"Checking performance with top {n} features...")

        # בחירת הפיצ'רים באמצעות RFE
        selector = RFE(estimator=estimator, n_features_to_select=n, step=1, verbose=1)
        selector.fit(train_df, train_target)
        selected_cols = train_df.columns[selector.support_].tolist()

        # הרצת Cross Validation עם שמירה של ה-Train Score
        cv_results = cross_validate(
            estimator,
            train_df[selected_cols],
            train_target,
            cv=5,
            scoring=scoring,
            return_train_score=True,  # כאן אנחנו מבקשים גם את הטריין
            n_jobs=-1,
            verbose= 3
        )

        # 3. איסוף כל הממוצעים (Train ו-Test)
        row = {
            'n_features': n,
            'Selected_Features': ", ".join(selected_cols)
        }

        # הוספת כל המדדים לדירוג
        for metric in scoring.keys():
            row[f'mean_train_{metric}'] = cv_results[f'train_{metric}'].mean()
            row[f'mean_test_{metric}'] = cv_results[f'test_{metric}'].mean()

        results.append(row)

    # 4. יצירת האקסל
    results_df = pd.DataFrame(results)

    # סידור עמודות שיהיה נוח לקרוא (קודם טסט ואז טריין)
    cols = ['n_features', 'Selected_Features'] + \
           [c for c in results_df.columns if 'mean_test' in c] + \
           [c for c in results_df.columns if 'mean_train' in c]

    file_name = f"{split_name}_Feature_Count_Comparison.xlsx"
    results_df[cols].to_excel(file_name, index=False)

    print(f"\n--- Results saved to: {file_name} ---")
    return results_df
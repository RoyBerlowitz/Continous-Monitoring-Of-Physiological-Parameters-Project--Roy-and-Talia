# כאן נוסיף את הפונקציות שמחשבות את הthreshold
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import cross_val_predict, StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score
from window_timing_translator_preprocessing import apply_smoothing
from joblib import Parallel, delayed

def print_metrics_table(y_true, y_pred, title): # לא לשכוח לעשות פה בלמטה את ההפרדה למקרים שיש צורך בmedian filter ושאין צורך
    """מחשב ומדפיס טבלת מדדים בצורה ברורה"""
    p = precision_score(y_true, y_pred, zero_division=0)
    s = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    a = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- {title} ---")
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 28)
    print(f"{'Precision':<15} | {p:.4f}")
    print(f"{'Sensitivity':<15} | {s:.4f}")
    print(f"{'Accuracy':<15} | {a:.4f}")
    print(f"{'F1 Score':<15} | {f1:.4f}")


def get_absolute_threshold_raw(y_true, y_probs):
    # this function gets the absolute best threshold in regard of F1 score, the threshold which will maximize the F1 score.
    # we get it on the raw data, without applying median filtering
    prec, rec, thresh = precision_recall_curve(y_true, y_probs)
    f1 = (2 * prec * rec) / (prec + rec + 1e-10)
    return thresh[np.argmax(f1[:-1])]


def get_threshold_median(X_sec, y_probs, y_true, window_size, random_threshold):
    groups = X_sec['recording_identifier'].values()
    preds = (y_probs >= random_threshold).astype(int)
    # בניית DataFrame זמני לצורך ה-Groupby
    temp_df = pd.DataFrame({
        'recording_identifier': groups,
        'prediction': preds
    })
    pred_df = apply_smoothing(temp_df, window_size)

    return f1_score(y_true, pred_df, zero_division=0)


# להתעסק בזה בהמשך כדי לעשות cross validation
# # חשוב לא לשכוח לשים group indicator = X_sec
# # להוסיף הסברים!!!!!
# def run_full_evaluation_pipeline(X_sec, y_sec, group_indicator, n_iteration =50, n_jobs = -1 ):
#     # this function is intended to help find the optimal threshold, the best working point in regard of maximizing the F1 score.
#     # as we want to have our model sensitive to the minority group but with high precision. F1 score takes into account both.
#     # we use group 5 folds cross validation logic in order to prevent data leakage and choose the threshold which is more suitable for the task the model faces.
#     print("Started choosing thresholds...")
#     cv = StratifiedGroupKFold(n_splits=5)
#     groups = group_indicator
#     y_probs = X_sec['weighted_prob'].values
#     # we preform the k folds
#     for  fold, (train_idx, val_idx) in enumerate(cv.split(X_sec, y_sec, groups)):
#         # we find the absolute thresholf for the x
#         get_absolute_threshold_raw(y_probs[train_idx], y_sec[val_idx])
#         # we find the absolute threshold for the median
#         random_thresholds = np.random.uniform(0.1, 0.9, size=n_iteration)
#         random_filter_sizes = np.random.choice([3, 5], size=n_iteration)
#         median_results = (Parallel(n_jobs=n_jobs)
#                    (delayed(get_threshold_median(X_sec, y_probs, y_sec, threshold, filter_size)
#                             for threshold, filter_size in zip(random_thresholds, random_filter_sizes))))
#         best_idx = np.argmax(median_results)
#         best_f1 = median_results[best_idx]
#         best_t = random_thresholds[best_idx]
#         best_filter_size = random_filter_sizes[best_idx]
#
#         print("\n" + "=" * 40)
#         print("      RANDOM SEARCH RESULTS      ")
#         print("=" * 40)
#         print(f"Best Filter Size: {best_filter_size}")
#         print(f"Best Threshold:   {best_t:.4f}")
#         print(f"Best F1 Score:    {best_f1:.4f}")
#         print("=" * 40)
#
#
#     # # הסרת עמודות מטא-דאטה לפני אימון
#     # X_train_clean = X_win.drop(columns=['recording_identifier', 'Handwashing time'])
#     #
#     # win_probs = cross_val_predict(model, X_train_clean, y_win,
#     #                               groups=groups, cv=cv, method='predict_proba')[:, 1]
#     #
#     # X_win_copy = X_win.copy()
#     # X_win_copy['window_probability'] = win_probs
#     #
#     # # 2. תרגום לשניות
#     # print("Step 2: Translating OOF Window Predictions to Seconds...")
#     # X_sec, y_sec = translate_to_seconds(X_win_copy)
#     #
#     # # 3. חישוב מדדים למצב RAW (ללא פילטר)
#     # print("Step 3: Optimizing RAW Threshold...")
#     # t_raw = get_absolute_threshold_raw(y_sec, X_sec['weighted_prob'])
#     # raw_final_preds = (X_sec['weighted_prob'] >= t_raw).astype(int)
#     # print_metrics_table(y_sec, raw_final_preds, f"RAW RESULTS (Threshold: {t_raw:.3f})")
#     #
#     # # 4. חישוב מדדים למצב SMOOTH (עם מדיאן פילטר)
#     # print("\nStep 4: Optimizing SMOOTH Threshold (with Median Filter)...")
#     # t_smooth = get_absolute_threshold_median(X_sec, y_sec, filter_size=3)
#     #
#     # # הפעלה סופית של הפילטר עם הסף שנמצא
#     # X_sec['temp_final'] = (X_sec['weighted_prob'] >= t_smooth).astype(int)
#     # smooth_final_preds = X_sec.groupby('recording_identifier')['temp_final'].transform(
#     #     lambda x: median_filter(x, size=3)
#     # )
#
#     print_metrics_table(y_sec, smooth_final_preds, f"SMOOTH RESULTS (Threshold: {t_smooth:.3f})")
#
#
#     return t_smooth, X_sec, y_sec
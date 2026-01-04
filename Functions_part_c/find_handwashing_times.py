import pandas as pd
import numpy as np
from .window_timing_translator_preprocessing import get_handwashing_times, apply_smoothing
from .timing_classifying_without_model import translate_prediction_into_time_point_prediction_with_weights, train_for_decision

# חשוב ליצור לפני זה את הclassification df גם לטסט וגם לטריין כולל הפרדיקשן - זה שלב מקדים!!!
# צריך להוסיף פה פונקציה שעושה אבליואציה ומחשבת את זה עבור הtest

def predict_times(train_df, test_df, data_files, classification_flag = "model", weight_flag = "Gaussian Weight"):
    # we obtain the handwashing seconds for each train and test df
    train_df = get_handwashing_times(train_df, data_files)
    test_df = get_handwashing_times(test_df, data_files)
    if classification_flag == "No model":
        train_x, train_y = translate_prediction_into_time_point_prediction_with_weights (train_df, weight_flag)
        y_probs = train_x["weighted_prob"]
        test_x, test_y = translate_prediction_into_time_point_prediction_with_weights (test_df, weight_flag)
        threshold_no_median, threshold_with_median, filter_size = train_for_decision(train_x, train_y, group_indicator =train_x["Group number"] , n_iteration=50, n_jobs=-1)
        pred_y_no_median_filter =  (y_probs >= threshold_no_median).astype(int)
        pred_y_with_median_filter =  (y_probs >= threshold_with_median).astype(int)
        smoothed_prediction = apply_smoothing(pred_y_with_median_filter, filter_size)
        # pred_y_no_median_filter, smoothed_prediction are our predictions - צריך להכין פונקצית אבליואציה למול הלייבלים האמיתיים ולקבל גם אולי סיווג לפי שניות ממש

        # כאן צריך להוסיף מטריקות אבליואציה וכו'


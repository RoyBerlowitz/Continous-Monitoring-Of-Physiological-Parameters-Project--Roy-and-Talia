import pandas as pd
import numpy as np
from .window_timing_translator_preprocessing import get_handwashing_times
from .timing_classifying_without_model import translate_prediction_into_time_point_prediction_with_weights

# חשוב ליצור לפני זה את הclassification df גם לטסט וגם לטריין כולל הפרדיקשן - זה שלב מקדים!!!
# צריך להוסיף פה פונקציה שעושה אבליואציה ומחשבת את זה עבור הtest

def predict_times(train_df, test_df, data_files, classification_flag = "model", weight_flag = "Gaussian Weight"):
    # we obtain the handwashing seconds for each train and test df
    train_df = get_handwashing_times(train_df, data_files)
    test_df = get_handwashing_times(test_df, data_files)
    if classification_flag == "No model":
        train_x, train_y = translate_prediction_into_time_point_prediction_with_weights (train_df, weight_flag)
        test_x, test_y = translate_prediction_into_time_point_prediction_with_weights (test_df, weight_flag)
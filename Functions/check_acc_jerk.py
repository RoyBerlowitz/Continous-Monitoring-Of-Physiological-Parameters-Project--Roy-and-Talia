import copy
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_selection import mutual_info_classif
from scipy.signal import savgol_filter

# ייבוא פונקציות העזר
from Functions.load_data import load_data
from extract_features_helper_functions import (
    safe_unwrap, compute_kurtosis, compute_skewness,
    apply_low_pass_filter, normalize_data, fix_baseline_wander,
    applying_windows, compute_signal_magnitude
)

# --- פונקציות חישוב נגזרות ---

def compute_derivatives_for_mi(data_list, window_size, sampling_rate=50, savigol_filter_flag=False):
    data_list = safe_unwrap(data_list)
    if data_list is None or len(data_list) <= window_size:
        return [np.nan] * 8

    dt = 1 / sampling_rate
    if not savigol_filter_flag:
        velocity = (data_list[window_size:] - data_list[:-window_size]) / (window_size * dt)
        acceleration = (velocity[window_size:] - velocity[:-window_size]) / (window_size * dt) if len(
            velocity) > window_size else np.array([np.nan])
    else:
        w_len = window_size if window_size % 2 != 0 else window_size + 1
        if w_len < 5: w_len = 5
        velocity = savgol_filter(data_list, window_length=w_len, polyorder=3, deriv=1, delta=dt)
        acceleration = savgol_filter(data_list, window_length=w_len, polyorder=3, deriv=2, delta=dt)

    return [np.nanstd(velocity), np.nanstd(acceleration), np.nanmedian(velocity), np.nanmedian(acceleration),
            compute_kurtosis(velocity), compute_kurtosis(acceleration), compute_skewness(velocity),
            compute_skewness(acceleration)]

def add_derivative_features_internal(df, column_list, window_size, use_savgol):
    suffixes = ['vel_std', 'acc_std', 'vel_med', 'acc_med', 'vel_kurt', 'acc_kurt', 'vel_skew', 'acc_skew']
    new_cols = {}
    for col in column_list:
        # בדיקה שהעמודה מכילה נתונים תקינים (לא NaN)
        res = df[col].apply(lambda x: pd.Series(
            compute_derivatives_for_mi(x, window_size, 50, use_savgol), index=suffixes))
        for s in suffixes:
            new_cols[f"{col}_{s}"] = res[s]
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)

# --- פונקציית ההרצה המרכזית ---

def run_jerk_mi_research():
    base_path = r'C:\Users\nirei\PycharmProjects\Continous monitoring'
    data_path = os.path.join(base_path, 'data')

    print("Step 1: Loading raw data files...")
    data_files = load_data(data_path)

    print("Step 2: Preparing X_matrix and y_vector...")
    df_segments = pd.read_pickle(os.path.join(base_path, 'segment_output.pkl'))
    X_matrix = df_segments[0].copy()
    y_vector = df_segments[1].values

    # תיקון קריטי 1: יצירת כל עמודות החיישנים (כולל Mag) מראש כדי למנוע KeyError
    sensors_cols = [
        'Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS',
        'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS',
        'Mag_X-AXIS', 'Mag_Y-AXIS', 'Mag_Z-AXIS'
    ]
    for col in sensors_cols:
        X_matrix[col] = None # שימוש ב-None מאפשר הכנסת רשימות/מערכים בקלות רבה יותר

    print("Step 3: Preprocessing and Windowing...")
    for recording in data_files.values():
        if 'Acc' not in recording: continue

        for sensor_name in ["Acc", "Gyro", "Mag"]:
            fs = 50 if sensor_name != "Mag" else 25
            unit = " (G)" if sensor_name == "Acc" else " (T)" if sensor_name == "Mag" else " (DEG/S)"

            for axis in ["X-AXIS", "Y-AXIS", "Z-AXIS"]:
                col_name = axis + unit
                if sensor_name in recording and col_name in recording[sensor_name]['data'].columns:
                    raw_data = recording[sensor_name]['data'][col_name].values
                    # Baseline wander
                    clean_data = fix_baseline_wander(raw_data, fs, filter_order=5, cutoff_frequency=0.5)
                    # Low Pass Filter
                    filtered_data = apply_low_pass_filter(clean_data, fs, filter_order=5, cutoff_frequency=10)
                    # Normalization
                    recording[sensor_name]['data'][col_name] = normalize_data(filtered_data)

        # הרצת חיתוך החלונות
        try:
            applying_windows(recording, X_matrix)
        except ValueError as e:
            # תיקון קריטי 2: התמודדות עם שגיאת ה-ndarray בהכנסת רשימות
            print(f"Handled a formatting issue during windowing for one recording.")
            continue

    # הסרת שורות שנותרו ריקות (אם יש כאלו)
    X_matrix.dropna(subset=['Acc_X-AXIS'], inplace=True)
    # עדכון ה-y_vector בהתאם לשורות שנותרו
    y_vector = y_vector[X_matrix.index]

    # הוספת SM (Signal Magnitude)
    print("Calculating Signal Magnitude...")
    X_features_base = X_matrix.copy()
    columns_for_mi = []
    for sensor in ["Acc", "Gyro"]:
        cols = [f"{sensor}_{ax}" for ax in ["X-AXIS", "Y-AXIS", "Z-AXIS"]]
        X_features_base[f"{sensor}_SM"] = X_features_base[cols].apply(compute_signal_magnitude, axis=1)
        columns_for_mi.extend(cols)
        columns_for_mi.append(f"{sensor}_SM")

    print("\nStep 4: Starting MI Research on Derivatives...")
    window_sizes = [1, 3,7, 15, 11, 21,31, 45]
    mi_results = []

    for w in window_sizes:
        for use_savgol in [False, True]:
            method = "Savitzky-Golay" if use_savgol else "Simple-Diff"
            print(f"  > Testing Window: {w}, Method: {method}")

            X_temp = X_features_base.copy()
            X_temp = add_derivative_features_internal(X_temp, columns_for_mi, w, use_savgol)

            derived_cols = [c for c in X_temp.columns if any(s in c for s in ['_vel_', '_acc_'])]

            current_mi = {"window": w, "method": method}
            for col in derived_cols:
                mask = X_temp[col].notna()
                if mask.sum() > 50:
                    score = mutual_info_classif(X_temp.loc[mask, [col]], y_vector[mask], random_state=42)[0]
                    current_mi[col] = score

            mi_results.append(current_mi)

    pd.DataFrame(mi_results).to_excel("Jerk_MI_Final_Results.xlsx", index=False)
    print("\nSuccess! Results saved.")

if __name__ == '__main__':
    run_jerk_mi_research()
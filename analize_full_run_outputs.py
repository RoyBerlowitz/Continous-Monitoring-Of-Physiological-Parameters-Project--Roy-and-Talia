from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle
import re
import os


from o02_train.Functions import WindowModelNames

def aggregate_run_outputs(
        base_path,
        model_name,
        mode='highest',
        target_run=None,
        exclude_model_rows=False,
        include_train=False,
        include_hyperparams=True,
        grouplen=None,
):
    """
    Aggregates Test results, (optional) Train results, and (optional) Hyperparameter pickles.
    """
    base_dir = Path(base_path)
    test_dfs, train_dfs, pkl_data = [], [], []

    folder_pattern = re.compile(r"^(?P<group>.+)_(?P<run>\d+)$")

    # 1. Directory Parsing
    run_list = []
    for entry in base_dir.iterdir():
        if entry.is_dir():
            match = folder_pattern.match(entry.name)
            if match:
                run_list.append({
                    'path': entry,
                    'group': match.group('group'),
                    'run_num': int(match.group('run')),
                    'grouplen': len(str(match.group('group')))
                })

    if not run_list:
        return "No valid folders found."

    df_runs = pd.DataFrame(run_list)

    # 2. Filtering Mode
    if mode == 'highest':
        df_runs = df_runs.sort_values('run_num').groupby('group').tail(1)
    elif mode == 'specific':
        if target_run is None:
            raise ValueError("Target run number required for 'specific' mode.")
        df_runs = df_runs[df_runs['run_num'] == int(target_run)]

    if grouplen:
        df_runs = df_runs[df_runs['grouplen'] == int(grouplen)]

    # 3. Process Excel and Pickle Files
    excel_name = f"{model_name}_second_model_evaluate_res.xlsx"
    pkl_name = f"choose_hyperparameters_{model_name}.pkl"

    for _, row in df_runs.iterrows():
        # Handle Excel Splits
        search_targets = [('test_run_outputs', test_dfs)]
        if include_train:
            search_targets.append(('train_run_outputs', train_dfs))

        for sub_dir, df_list in search_targets:
            file_path = row['path'] / sub_dir / excel_name
            if file_path.exists():
                df = pd.read_excel(file_path)
                df['group_name'] = row['group']
                df['run_num'] = row['run_num']
                df['group_model'] = row['group'] + df['model_name']

                if exclude_model_rows and 'model_name' in df.columns:
                    df = df[~df['model_name'].astype(str).str.startswith(model_name)]
                df_list.append(df)

        # Handle Pickle Files
        if include_hyperparams:
            pkl_path = row['path'] / 'train_pkls' / pkl_name
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    # If data is a dict, wrap in list for DF; if list of dicts, it's ready
                    temp_df = pd.json_normalize(data) if isinstance(data, dict) else pd.DataFrame(data)
                    temp_df['group_name'] = row['group']
                    temp_df['run_num'] = row['run_num']
                    pkl_data.append(temp_df)

    # 4. Final Exports
    metrics = ["precision", "specificity", "sensitivity", "accuracy", "f1_score", "cohen_kappa",]
    results = []
    now = datetime.now()
    formatted_time = now.strftime("%m-%d_%Hh%Mm%Ssec")
    folder_path = Path(__file__).resolve().parent / "analysis_full_run_model_outputs"
    os.makedirs(folder_path, exist_ok=True)

    if test_dfs:
        test_dfs = pd.concat(test_dfs, ignore_index=True)
        test_dfs.to_excel(f"{folder_path}/agg_test_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Test Excel created")

        summary_df = (test_dfs.groupby("model_name")[metrics].agg(["mean", "std"]))
        summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
        summary_df.to_excel(f"{folder_path}/metrics_test_{mode}_{formatted_time}.xlsx")
        results.append("Test metrics Excel created")

    if train_dfs:
        train_dfs = pd.concat(train_dfs, ignore_index=True)
        train_dfs.to_excel(f"{folder_path}/agg_train_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Train Excel created")

        summary_df = (train_dfs.groupby("model_name")[metrics].agg(["mean", "std"]))
        summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
        summary_df.to_excel(f"{folder_path}/metrics_train_{mode}_{formatted_time}.xlsx")
        results.append("Train metrics Excel created")

    if pkl_data:
        pd.concat(pkl_data, ignore_index=True).to_excel(f"{folder_path}/agg_hyperparams_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Hyperparameters Excel created")

    print(" | ".join(results) if results else "No data found.")
    return


def aggregate_run_outputs(
        base_path,
        model_name,
        mode='highest',
        target_run=None,
        exclude_model_rows=False,
        include_train=False,
        include_hyperparams=True,
        grouplen=None,
):
    """
    Aggregates Test results, (optional) Train results, and Top 3 Hyperparameters based on PRC.
    Robust to filename typos (Forest/Forrest).
    """
    base_dir = Path(base_path)
    test_dfs, train_dfs, hp_data = [], [], []

    folder_pattern = re.compile(r"^(?P<group>.+)_(?P<run>\d+)$")

    # 1. Directory Parsing
    run_list = []
    # בדיקה שהתיקייה הראשית קיימת
    if not base_dir.exists():
        print(f"Error: Base path does not exist: {base_dir}")
        return

    for entry in base_dir.iterdir():
        if entry.is_dir():
            match = folder_pattern.match(entry.name)
            if match:
                run_list.append({
                    'path': entry,
                    'group': match.group('group'),
                    'run_num': int(match.group('run')),
                    'grouplen': len(str(match.group('group')))
                })

    if not run_list:
        print("No valid run folders found.")
        return

    df_runs = pd.DataFrame(run_list)

    # 2. Filtering Mode
    if mode == 'highest':
        df_runs = df_runs.sort_values('run_num').groupby('group').tail(1)
    elif mode == 'specific':
        if target_run is None:
            raise ValueError("Target run number required for 'specific' mode.")
        df_runs = df_runs[df_runs['run_num'] == int(target_run)]

    if grouplen:
        df_runs = df_runs[df_runs['grouplen'] == int(grouplen)]

    print(f"Found {len(df_runs)} runs to process.")  # Debug

    # 3. Process Excel Files
    eval_excel_name = f"{model_name}_second_model_evaluate_res.xlsx"

    # שמות אפשריים לקובץ ההייפר-פרמטרים (עם ובלי שגיאת הכתיב)
    possible_hp_names = [
        f"find_best_hp_{model_name}_Search_Results.xlsx",  # Standard
        f"find_best_hp_{model_name.replace('Forest', 'Forrest')}_Search_Results.xlsx",  # Typo fix
        f"choose_hyperparameters_{model_name}.xlsx"  # Old format backup
    ]

    for _, row in df_runs.iterrows():
        # --- Evaluation Results (Test/Train) ---
        search_targets = [('test_run_outputs', test_dfs)]
        if include_train:
            search_targets.append(('train_run_outputs', train_dfs))

        for sub_dir, df_list in search_targets:
            file_path = row['path'] / sub_dir / eval_excel_name
            if file_path.exists():
                try:
                    df = pd.read_excel(file_path)
                    df['group_name'] = row['group']
                    df['run_num'] = row['run_num']
                    df['group_model'] = row['group'] + df['model_name']

                    if exclude_model_rows and 'model_name' in df.columns:
                        df = df[~df['model_name'].astype(str).str.startswith(model_name)]
                    df_list.append(df)
                except Exception as e:
                    print(f"Error reading eval excel in {row['group']}: {e}")

        # --- Hyperparameters (Top 3 based on PRC) ---
        if include_hyperparams:
            hp_path = None
            # מנסה למצוא את הקובץ באחד השמות האפשריים
            for name in possible_hp_names:
                temp_path = row['path'] / 'train_run_outputs' / name
                if temp_path.exists():
                    hp_path = temp_path
                    break

            if hp_path:
                try:
                    hp_df = pd.read_excel(hp_path)

                    target_metric = 'mean_test_PRC'

                    if target_metric in hp_df.columns:
                        # 1. מיון יורד לפי PRC
                        top_3_df = hp_df.sort_values(target_metric, ascending=False).head(3).copy()

                        # הוספת מטא-דאטה
                        top_3_df['group_name'] = row['group']
                        top_3_df['run_num'] = row['run_num']
                        top_3_df['rank'] = range(1, len(top_3_df) + 1)

                        # שמירת העמודות החשובות + מה שקיים
                        base_cols = ['group_name', 'run_num', 'rank', 'params', target_metric]
                        extra_cols = [c for c in ['mean_test_AUC', 'mean_test_F1', 'mean_train_PRC'] if
                                      c in top_3_df.columns]

                        hp_data.append(top_3_df[base_cols + extra_cols])
                    else:
                        print(f"Warning: '{target_metric}' column missing in {hp_path.name} for group {row['group']}")

                except Exception as e:
                    print(f"Error reading HP excel for group {row['group']}: {e}")
            else:
                # הדפסה שתעזור להבין איפה הקובץ חסר
                print(
                    f"DEBUG: HP file not found for group {row['group']}. Checked in: {row['path'] / 'train_run_outputs'}")

    # 4. Final Exports
    metrics = ["precision", "specificity", "sensitivity", "accuracy", "f1_score", "cohen_kappa", ]
    results = []
    now = datetime.now()
    formatted_time = now.strftime("%m-%d_%Hh%Mm%Ssec")
    folder_path = Path(__file__).resolve().parent / "analysis_full_run_model_outputs"
    os.makedirs(folder_path, exist_ok=True)

    if test_dfs:
        test_dfs = pd.concat(test_dfs, ignore_index=True)
        test_dfs.to_excel(f"{folder_path}/agg_test_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Test Excel created")

        valid_metrics = [m for m in metrics if m in test_dfs.columns]
        if valid_metrics:
            summary_df = (test_dfs.groupby("model_name")[valid_metrics].agg(["mean", "std"]))
            summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
            summary_df.to_excel(f"{folder_path}/metrics_test_{mode}_{formatted_time}.xlsx")
            results.append("Test metrics Excel created")

    if train_dfs:
        train_dfs = pd.concat(train_dfs, ignore_index=True)
        train_dfs.to_excel(f"{folder_path}/agg_train_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Train Excel created")

    if hp_data:
        # שמירת קובץ הטופ 3 המאוחד
        pd.concat(hp_data, ignore_index=True).to_excel(f"{folder_path}/agg_top3_HP_PRC_{mode}_{formatted_time}.xlsx",
                                                       index=False)
        results.append("Top 3 Hyperparameters (PRC) Excel created")
    else:
        print("Warning: No Hyperparameter data was collected!")

    print(" | ".join(results) if results else "No data found.")
    return


def aggregate_run_outputs(
        base_path,
        model_name,
        mode='highest',
        target_run=None,
        exclude_model_rows=False,
        include_train=False,
        include_hyperparams=True,
        include_rfecv=True,
        include_features=True,
        grouplen=None,
):
    """
    Aggregates Test results, Top 3 Hyperparameters, Top 3 RFECV counts,
    and the actual List of Selected Features from the pickle.
    Robust to filename case sensitivity, typos, and nested list structures in pickles.
    """
    base_dir = Path(base_path)
    test_dfs, train_dfs, hp_data, rfecv_data, features_data = [], [], [], [], []

    folder_pattern = re.compile(r"^(?P<group>.+)_(?P<run>\d+)$")

    # 1. Directory Parsing
    run_list = []
    if not base_dir.exists():
        print(f"Error: Base path does not exist: {base_dir}")
        return

    for entry in base_dir.iterdir():
        if entry.is_dir():
            match = folder_pattern.match(entry.name)
            if match:
                run_list.append({
                    'path': entry,
                    'group': match.group('group'),
                    'run_num': int(match.group('run')),
                    'grouplen': len(str(match.group('group')))
                })

    if not run_list:
        print("No valid run folders found.")
        return

    df_runs = pd.DataFrame(run_list)

    # 2. Filtering Mode
    if mode == 'highest':
        df_runs = df_runs.sort_values('run_num').groupby('group').tail(1)
    elif mode == 'specific':
        if target_run is None:
            raise ValueError("Target run number required for 'specific' mode.")
        df_runs = df_runs[df_runs['run_num'] == int(target_run)]

    if grouplen:
        df_runs = df_runs[df_runs['grouplen'] == int(grouplen)]

    print(f"Found {len(df_runs)} runs to process.")

    # 3. Process Files
    eval_excel_name = f"{model_name}_second_model_evaluate_res.xlsx"

    for _, row in df_runs.iterrows():
        train_output_dir = row['path'] / 'train_run_outputs'

        # --- A. Evaluation Results (Test/Train) ---
        search_targets = [('test_run_outputs', test_dfs)]
        if include_train:
            search_targets.append(('train_run_outputs', train_dfs))

        for sub_dir, df_list in search_targets:
            file_path = row['path'] / sub_dir / eval_excel_name
            # Smart find for evaluate file
            if not file_path.exists():
                dir_path = row['path'] / sub_dir
                if dir_path.exists():
                    for f in dir_path.iterdir():
                        if f.name.lower() == eval_excel_name.lower():
                            file_path = f
                            break

            if file_path.exists():
                try:
                    df = pd.read_excel(file_path)
                    df['group_name'] = row['group']
                    df['run_num'] = row['run_num']
                    df['group_model'] = row['group'] + df['model_name']
                    if exclude_model_rows and 'model_name' in df.columns:
                        df = df[~df['model_name'].astype(str).str.startswith(model_name)]
                    df_list.append(df)
                except Exception as e:
                    print(f"Error reading eval excel in {row['group']}: {e}")

        # --- B. Hyperparameters (Top 3 based on PRC) ---
        if include_hyperparams:
            hp_path = None
            if train_output_dir.exists():
                for file in train_output_dir.iterdir():
                    fname = file.name.lower()
                    if "find_best_hp_" in fname and "search_results" in fname and fname.endswith(".xlsx"):
                        hp_path = file
                        break
            if hp_path:
                try:
                    hp_df = pd.read_excel(hp_path)
                    target_metric = 'mean_test_PRC'
                    if target_metric in hp_df.columns:
                        top_3_df = hp_df.sort_values(target_metric, ascending=False).head(3).copy()
                        top_3_df['group_name'] = row['group']
                        top_3_df['run_num'] = row['run_num']
                        top_3_df['rank'] = range(1, len(top_3_df) + 1)
                        base_cols = ['group_name', 'run_num', 'rank', 'params', target_metric]
                        extra_cols = [c for c in ['mean_test_AUC', 'mean_test_F1', 'mean_train_PRC'] if
                                      c in top_3_df.columns]
                        hp_data.append(top_3_df[base_cols + extra_cols])
                except Exception as e:
                    print(f"Error reading HP excel for group {row['group']}: {e}")

        # --- C. RFECV Feature Count (Top 3 based on PRC) ---
        if include_rfecv:
            rfecv_path = None
            if train_output_dir.exists():
                for file in train_output_dir.iterdir():
                    fname = file.name.lower()
                    if "rfecv" in fname and "performance" in fname and fname.endswith(".xlsx"):
                        rfecv_path = file
                        break
            if rfecv_path:
                try:
                    rfe_df = pd.read_excel(rfecv_path)
                    if 'mean_test_PRC' in rfe_df.columns and 'n_features' in rfe_df.columns:
                        top_3_rfe = rfe_df.sort_values('mean_test_PRC', ascending=False).head(3).copy()
                        top_3_rfe['group_name'] = row['group']
                        top_3_rfe['run_num'] = row['run_num']
                        top_3_rfe['rank'] = range(1, len(top_3_rfe) + 1)
                        rfecv_data.append(top_3_rfe)
                except Exception as e:
                    print(f"Error reading RFECV excel for group {row['group']}: {e}")

        # --- D. Selected Features List (From Pickle) - FIXED ---
        if include_features:
            features_pkl_name = "select_features_random_forest.pkl"

            # חיפוש ב-train_run_outputs
            feat_path = row['path'] / 'train_run_outputs' / features_pkl_name
            # אם לא שם, חיפוש ב-train_pkls
            if not feat_path.exists():
                feat_path = row['path'] / 'train_pkls' / features_pkl_name

            if feat_path.exists():
                try:
                    with open(feat_path, 'rb') as f:
                        selected_feats = pickle.load(f)

                    # 1. טיפול ב-Numpy Array
                    if hasattr(selected_feats, 'tolist'):
                        selected_feats = selected_feats.tolist()

                    # 2. טיפול ברשימה בתוך רשימה (Flattening)
                    # התיקון הקריטי: אם זה [['a','b']] נהפוך ל-['a','b']
                    if isinstance(selected_feats, list) and len(selected_feats) > 0:
                        # בדיקה אם האיבר הראשון הוא רשימה/טופל
                        if isinstance(selected_feats[0], (list, tuple)):
                            flat_list = []
                            for sublist in selected_feats:
                                if isinstance(sublist, str):
                                    flat_list.append(sublist)
                                else:
                                    try:
                                        flat_list.extend(sublist)
                                    except TypeError:
                                        flat_list.append(str(sublist))
                            selected_feats = flat_list

                    # 3. המרה סופית למחרוזת (Safe Join)
                    feats_str = ", ".join([str(f) for f in selected_feats])
                    feat_count = len(selected_feats)

                    features_data.append({
                        'group_name': row['group'],
                        'run_num': row['run_num'],
                        'n_selected': feat_count,
                        'feature_list': feats_str
                    })
                except Exception as e:
                    print(f"Error reading features pickle for group {row['group']}: {e}")
            else:
                pass

                # 4. Final Exports
    metrics = ["precision", "specificity", "sensitivity", "accuracy", "f1_score", "cohen_kappa", ]
    results = []
    now = datetime.now()
    formatted_time = now.strftime("%m-%d_%Hh%Mm%Ssec")
    folder_path = Path(__file__).resolve().parent / "analysis_full_run_model_outputs"
    os.makedirs(folder_path, exist_ok=True)

    if test_dfs:
        test_dfs = pd.concat(test_dfs, ignore_index=True)
        test_dfs.to_excel(f"{folder_path}/agg_test_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Test Excel created")

        valid_metrics = [m for m in metrics if m in test_dfs.columns]
        if valid_metrics:
            summary_df = (test_dfs.groupby("model_name")[valid_metrics].agg(["mean", "std"]))
            summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
            summary_df.to_excel(f"{folder_path}/metrics_test_{mode}_{formatted_time}.xlsx")
            results.append("Test metrics Excel created")

    if train_dfs:
        train_dfs = pd.concat(train_dfs, ignore_index=True)
        train_dfs.to_excel(f"{folder_path}/agg_train_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Train Excel created")

    if hp_data:
        pd.concat(hp_data, ignore_index=True).to_excel(f"{folder_path}/agg_top3_HP_PRC_{mode}_{formatted_time}.xlsx",
                                                       index=False)
        results.append("Top 3 Hyperparameters (PRC) Excel created")

    if rfecv_data:
        pd.concat(rfecv_data, ignore_index=True).to_excel(f"{folder_path}/agg_top3_RFECV_{mode}_{formatted_time}.xlsx",
                                                          index=False)
        results.append("Top 3 RFECV Features Excel created")

    if features_data:
        pd.DataFrame(features_data).to_excel(f"{folder_path}/agg_best_features_list_{mode}_{formatted_time}.xlsx",
                                             index=False)
        results.append("Best Features List Excel created")

    print(" | ".join(results) if results else "No data found.")
    return
if __name__ == "__main__":
    full_run_model_outputs_path = Path(__file__).resolve().parent / "full_run_model_outputs"
    #window_model = WindowModelNames.XGBOOST
    window_model = WindowModelNames.RANDOM_FOREST
    grouplen = 2 #2=big groups 3=individual None=all
    print(f'Running for model {window_model} and group name length of {grouplen}')
    aggregate_run_outputs(full_run_model_outputs_path, window_model, exclude_model_rows=True, grouplen=grouplen)#, target_run=1, mode='specific' )

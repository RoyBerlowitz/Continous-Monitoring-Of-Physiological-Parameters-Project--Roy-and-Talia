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
    results = []
    now = datetime.now()
    formatted_time = now.strftime("%m-%d_%Hh%Mm%Ssec")
    folder_path = Path(__file__).resolve().parent / "analysis_full_run_model_outputs"
    os.makedirs(folder_path, exist_ok=True)

    if test_dfs:
        pd.concat(test_dfs, ignore_index=True).to_excel(f"{folder_path}/agg_test_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Test Excel created")

    if train_dfs:
        pd.concat(train_dfs, ignore_index=True).to_excel(f"{folder_path}/agg_train_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Train Excel created")

    if pkl_data:
        pd.concat(pkl_data, ignore_index=True).to_excel(f"{folder_path}/agg_hyperparams_{mode}_{formatted_time}.xlsx", index=False)
        results.append("Hyperparameters Excel created")

    print(" | ".join(results) if results else "No data found.")
    return

if __name__ == "__main__":
    full_run_model_outputs_path = Path(__file__).resolve().parent / "full_run_model_outputs"
    window_model = WindowModelNames.XGBOOST
    grouplen = 2 #2=big groups 3=individual None=all
    print(f'Running for model {window_model} and group name length of {grouplen}')
    aggregate_run_outputs(full_run_model_outputs_path, window_model, exclude_model_rows=True, grouplen=grouplen)#, target_run=1, mode='specific' )

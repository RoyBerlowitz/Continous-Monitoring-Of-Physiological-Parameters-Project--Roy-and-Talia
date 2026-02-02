from pathlib import Path
import shutil

from .extract_features_helper_functions import get_cnn_embeddings

def cnn_embedding(X_matrix, y_vec, group_name='', test_flag=False):
    # creating an embedder
    columns_names = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS',
                     'Mag_X-AXIS', 'Mag_Y-AXIS', 'Mag_Z-AXIS', 'Acc_SM', 'Mag_SM', 'Gyro_SM']

    columns_names_for_embedding = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS',
                                   'Gyro_Z-AXIS', ]

    group_indicator = X_matrix['Group number'].astype(str) + "_" + X_matrix['Participant ID'].astype(str)

    other_params = {"num_epochs":2} if test_flag else {"num_epochs":30,"batch_size":64,"dropout":0.25,"steps":8}

    model_path = Path(__file__).resolve().parent.parent.parent / "run_outputs" / f'{group_name}cnn_train_weights.pth'

    X_matrix = get_cnn_embeddings(X_matrix,
                                  target=y_vec,
                                  group_col="Group number + Participant ID",
                                  group_indicator =  group_indicator,
                                  column_list=columns_names_for_embedding,
                                  test_flag=test_flag,
                                  model_path=model_path,
                                  embedding_size=16,
                                  batch_size=64,
                                  **other_params
                                  )

    if not test_flag:
        #save model into test folder pkls
        model_path_test = Path(__file__).resolve().parent.parent.parent.parent / "o02_test" / "pkls" / f'{group_name}cnn_train_weights.pth'
        shutil.copy2(model_path, model_path_test)
        print(f"Copied from {model_path} to {model_path_test}")

    # getting rid of the columns with the vectors of values
    X_matrix = X_matrix.drop(labels=columns_names, axis=1)

    administrative_features = ['Split_ID', 'First second of the activity', 'Last second of the activity',
                               'Participant ID', 'Group number', 'Recording number', 'Protocol']

    informative_features = [
        "cnn_emb_7",
        "Acc_X-AXIS_acceleration_std",
        "cnn_emb_2",
        "Acc_X_Z_CORR",
        "cnn_emb_13",
        "cnn_emb_0",
        "Acc_Z-AXIS_CUSUM+_Feature",
        "cnn_emb_10",
        "Acc_Z-AXIS_CUSUM-_Feature",
        "Gyro_Z-AXIS_AbsCV",
        "Acc_SM_acceleration_median",
        "Gyro_SM_velocity_median",
        "Gyro_Y-AXIS_velocity_median",
        "Mag_Y-AXIS_median",
        "Acc_X-AXIS_velocity_skewness",
        "Mag_MEAN_AXES_CORR",
        "cnn_emb_6",
        "Gyro_X-AXIS_CUSUM-_Feature",
        "Gyro_SM_acceleration_kurtosis",
        "Acc_Z-AXIS_velocity_skewness"
    ]
    features_to_keep = administrative_features + informative_features
    X_matrix = X_matrix[features_to_keep]

    return X_matrix
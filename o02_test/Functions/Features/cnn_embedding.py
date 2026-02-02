from pathlib import Path
import shutil

from .extract_features_helper_functions import get_cnn_embeddings

def cnn_embedding_full_workflow(X_matrix, y_vec, informative_features, group_name='', test_flag=False):
    # creating an embedder
    columns_names = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS',
                     'Mag_X-AXIS', 'Mag_Y-AXIS', 'Mag_Z-AXIS', 'Acc_SM', 'Mag_SM', 'Gyro_SM']

    columns_names_for_embedding = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS',
                                   'Gyro_Z-AXIS']

    group_indicator = X_matrix['Group number'].astype(str) + "_" + X_matrix['Participant ID'].astype(str)

    other_params = {"num_epochs":2} if test_flag else {"num_epochs":30,"dropout":0.25,"steps":8}

    model_path = Path(__file__).resolve().parent.parent.parent / "run_outputs" / f'{group_name}cnn_train_weights.pth'

    X_matrix = get_cnn_embeddings(X_matrix,
                                  target=y_vec,
                                  group_col="Group number + Participant ID",
                                  group_indicator = group_indicator,
                                  column_list=columns_names_for_embedding,
                                  test_flag=test_flag,
                                  model_path=model_path,
                                  embedding_size=16,
                                  batch_size=64,
                                  num_epochs=30,
                                  dropout=0.3,
                                  )

    if not test_flag:
        #save model into test folder pkls
        model_path_test = Path(__file__).resolve().parent.parent.parent.parent / "o02_test" / "pkls" / f'{group_name}cnn_train_weights.pth'
        shutil.copy2(model_path, model_path_test)
        print(f"Copied from {model_path} to {model_path_test}")

    # getting rid of the columns with the vectors of values
    X_matrix = X_matrix.drop(labels=columns_names, axis=1)

    administrative_features = ['First second of the activity', 'Last second of the activity',
                               'Participant ID', 'Group number', 'Recording number', 'Protocol']

    selected_feats = informative_features + administrative_features
    columns_to_keep = [c in X_matrix.columns for c in selected_feats]
    X_matrix = X_matrix[columns_to_keep]

    return X_matrix
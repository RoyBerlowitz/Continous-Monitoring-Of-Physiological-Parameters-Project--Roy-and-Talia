from pathlib import Path
import shutil

from .extract_features_helper_functions import get_cnn_embeddings

def cnn_embedding_full_workflow(X_matrix, y_vec, group_name='', test_flag=False):
    # creating an embedder
    columns_names = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS',
                     'Mag_X-AXIS', 'Mag_Y-AXIS', 'Mag_Z-AXIS', 'Acc_SM', 'Mag_SM', 'Gyro_SM']

    columns_names_for_embedding = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS',
                                   'Gyro_Z-AXIS']

    # group indicator for the seperation of val and test
    group_indicator = X_matrix['Group number'].astype(str) + "_" + X_matrix['Participant ID'].astype(str)

    # the path in which the CNN model weights will be saved
    model_path = Path(__file__).resolve().parent.parent.parent / "run_outputs" / f'{group_name}cnn_train_weights.pth'
    print(model_path)

    # we use the CNN embedding function
    X_matrix = get_cnn_embeddings(X_matrix,
                                  target=y_vec,
                                  group_col="Group number + Participant ID",
                                  group_indicator = group_indicator,
                                  column_list=columns_names_for_embedding,
                                  test_flag=True,#test_flag,
                                  model_path=model_path,
                                  embedding_size=16,
                                  batch_size=64,
                                  num_epochs=30,
                                  dropout=0.3,
                                  )

    if not test_flag:
        #save model into test folder pkls
        model_path_test = Path(__file__).resolve().parent.parent.parent.parent / "02_test" / "run_outputs" / f'{group_name}cnn_train_weights.pth'
        shutil.copy2(model_path, model_path_test)
        print(f"Copied from {model_path} to {model_path_test}")

    # getting rid of the columns with the vectors of values
    X_matrix = X_matrix.drop(labels=columns_names, axis=1)

    # Now we want to remove columns in which all the values are zeros, as they won't contribute and may damage the feature vetting
    cols_to_drop = (X_matrix == 0).all()
    zero_cols = X_matrix.columns[cols_to_drop]

    # we clean the zero columns
    X_matrix = X_matrix.drop(columns=zero_cols)
    return X_matrix
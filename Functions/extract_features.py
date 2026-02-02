import copy

import matplotlib.pyplot as plt
import numpy as np

from .extract_features_helper_functions import *
from sklearn.feature_selection import mutual_info_classif


##-------Main function - extract features-------##

def extract_features (X_matrix , Y_vector, data_files, more_prints, test_flag = False, group_name = 2):
    if more_prints: print ("extracting features again ...")

    num_features = 0

    #defining the names of columns in which the data of each window is going to be saved into
    columns_names = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS',
                     'Mag_X-AXIS', 'Mag_Y-AXIS', 'Mag_Z-AXIS']
    for col in columns_names:
        X_matrix[col] = pd.Series(dtype=object)

    # We preform the loop on all the recording to do baseline wander correction on the data and then apply the windows
    for recording in data_files.values():
        #adding the window's data for the suitable row
        for sensor_name in ["Acc", "Gyro", "Mag"]:
            sampling_frequency = 50
            if sensor_name == "Acc":
                unit =" (G)"
            elif sensor_name == "Mag":
                unit =" (T)"
                sampling_frequency = 25 #As the magnetometer got different sampling frequency
            elif sensor_name == "Gyro":
                unit =" (deg/s)".upper()
            for axis in ["X-AXIS", "Y-AXIS", "Z-AXIS"]:
                axis_data = recording[sensor_name]['data'][axis + unit].values
                #Conductiong baseline wander on the entire data from a certain recording in a certain sensor
                new_axis_data = fix_baseline_wander (axis_data, sampling_frequency, filter_order =5 , cutoff_frequency = 0.5)
                recording[sensor_name]['data'][axis + unit] = new_axis_data



        # We use the pre-defined function in order to get the data for each window from each recording
        applying_windows(recording, X_matrix)

    X_features = copy.deepcopy(X_matrix) #creating X feature before feature extraction

    for sensor_name in ["Acc", "Gyro", "Mag"]:
        X_features[sensor_name + "_SM"] = X_features[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(compute_signal_magnitude,axis=1)
        columns_names.append(sensor_name + "_SM")

    #At this point, the frequency domain features will be extracted.
    X_features, num_features = add_frequency_domain_features(X_features, columns_names, num_features, more_prints)

    #Normalization may change frequency domain behaviour, as it includes scaling.

    #We extract statistical metrics - Kurtosis and Skewness - our article-based features - we do it because we want to see how data behaves and normalization may change it.
    #Their formulas standardize the data so we give them the 'raw' data
    X_features, num_features = add_disribution_features(X_features, columns_names, num_features, more_prints)
    #Thus, normalization will be conducted later

    #we go over the recording to get the normalize data for the entire axis
    for recording in data_files.values():
        #adding the window's data for the suitable row
        for sensor_name in ["Acc", "Gyro", "Mag"]:
            if sensor_name == "Acc":
                unit =" (G)"
                dt = 1/50
            elif sensor_name == "Mag":
                unit =" (T)"
                dt = 1/25
            elif sensor_name == "Gyro":
                unit =" (deg/s)".upper()
                dt = 1/50
            for axis in ["X-AXIS", "Y-AXIS", "Z-AXIS"]:
                axis_data = recording[sensor_name]['data'][axis + unit].values
                #Conductiong LPF on the entire data from a certain recording in a certain sensor
                #new_axis_data = apply_low_pass_filter (axis_data, sampling_frequency, filter_order =5 , cutoff_frequency = 10)
                new_axis_data = axis_data
                # Conducting normalization on the entire data from a certain recording in a certain sensor
                normalized_axis_data = normalize_data(new_axis_data)
                recording[sensor_name]['data'][axis + unit] = normalized_axis_data

    #Now, let's find the magnitude again, this time normalized.
    for sensor_name in ["Acc", "Gyro", "Mag"]:
        X_features[sensor_name + "_SM"] = X_features[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(compute_signal_magnitude,axis=1)


    #We extract the basic metrics - STD, mean, median, max, min, peak-to-peak difference, RMS, zero-crossing, IQR
    X_features,num_features = add_basic_metrics(X_features, columns_names, num_features, more_prints)

    #We extract the SR, Area Under Graph, etc
    X_features, num_features = add_time_dependent_features(X_features, columns_names, num_features)


    # Adding derivative-oriented features - the kurtosis, median and std of the velocity and acceleration in each window for Gyro and Acc
    X_features, num_features = add_derivative_features(X_features, columns_names, num_features, more_prints)

    #We find the COSUM feature - we do it only for  the Acc and Gyro as the changes in Mag is much less trackable and significant
    for column in columns_names:
        if not "Mag" in column:
            X_features = add_Cosum_metrics(X_features, column, more_prints)
            num_features += 4
            if "SM" in column:
                num_features -= 1

    # #getting the embedding vector by a trained cnn
    # columns_names_for_embedding = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS']
    #group_name = '0'+ str(group_name) + '_'
    #group_indicator = X_features['Group number'].astype(str) + "_" + X_features['Participant ID'].astype(str)
    # X_features = get_cnn_embeddings(X_features,
    #                    target= Y_vector,
    #                    group_col = "Group number + Participant ID",
    #                    group_indicator =  group_indicator,
    #                    column_list = columns_names_for_embedding,
    #                    test_flag=test_flag,
    #                    model_path=group_name+'cnn_weights.pth',
    #                    embedding_size=16,
    #                    num_epochs=30,
    #                    batch_size=64,
    #                    dropout= 0.3)
    # num_features += 16
    # # getting rid of the columns with the vectors of values
    # X_features = X_features.drop(labels=columns_names, axis=1)


    #Now we want to remove columns in which all the values are zeros, as they won't contribute and may damage the feature vetting
    # cols_to_drop = (X_features == 0).all()
    # zero_cols = X_features.columns[cols_to_drop]
    #
    # # we clean the zero columns
    # if more_prints: print(f"added {num_features - len(zero_cols)} columns")
    # X_features = X_features.drop(columns=zero_cols)



    return X_features


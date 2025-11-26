import copy
from operator import index

import PyEMD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
import os
from scipy.signal import butter, filtfilt

##As loading data happens in both Part A and Part B, we define a function named Load data.
#This function get data path and return dict that include all of the CSV data of recording from all sensor, alongside the identifiers
def load_data(data_path):
    all_items = os.listdir(data_path)
    data_files = {}
    for file_path in all_items:
        #This implementation is done to make sure no data will be missed to end of CSV instead of csv
        if file_path.lower().endswith('.csv') and os.path.isfile(os.path.join(data_path, file_path)):
            file_name = os.path.basename(file_path)

            # Here, we create for each record a dict that includes the Recording number, the group number, Participant ID
            # and another identifier for the record type which is a dict of the actual data and the record type.
            #this will help differntiate between data_types
            recording_identifier = file_name[:7]
            list_of_elements = file_name.split('_')
            recording_type = list_of_elements[3].removesuffix(".csv")
            psv_data = pd.read_csv(os.path.join(data_path, file_path))
            psv_data.columns = psv_data.columns.str.upper()
            psv_data = psv_data.dropna()

            data_dict = {"Recording type": recording_type, "data": psv_data}

            if data_files.get(recording_identifier):
                data_files[recording_identifier][recording_type] = data_dict
            else:
                data_files[recording_identifier] = {}
                data_files[recording_identifier]["Group number"] = list_of_elements[0]
                data_files[recording_identifier]["Recording number"] = list_of_elements[1]
                data_files[recording_identifier]["Participant ID"] = list_of_elements[2]
                data_files[recording_identifier][recording_type] = data_dict
                # A list that will get the time of handwashing. It is a list in case there are more than one handwashing events in the recording
                data_files[recording_identifier]['Handwashing time'] = []
                #the next one will help us to distinguish between protocol recordings to regular one
                data_files[recording_identifier]['Protocol'] = 0
    return data_files

##-------Part A: Segmentation-------

# Window step is the delay between each window. Because the

def segment_signal (data_path, window_size, window_step):
    data_files = load_data(data_path)
    print(data_files["02_01_A"]["Acc"]["data"])
    # all_items = os.listdir(data_path)
    # data_files = {}
    # for file_path in all_items:
    #     #This implementation is done to make sure no data will be missed to end of CSV instead of csv
    #     if file_path.lower().endswith('.csv') and os.path.isfile(os.path.join(data_path, file_path)):
    #         file_name = os.path.basename(file_path)
    #
    #         # Here, we create for each record a dict that includes the Recording number, the group number, Participant ID
    #         # and another identifier for the record type which is a dict of the actual data and the record type.
    #         #this will help differntiate between data_types
    #         recording_identifier = file_name[:7]
    #         list_of_elements = file_name.split('_')
    #         recording_type = list_of_elements[3].removesuffix(".csv")
    #         psv_data = pd.read_csv(os.path.join(data_path, file_path))
    #         psv_data.columns = psv_data.columns.str.upper()
    #         psv_data = psv_data.dropna()
    #
    #         data_dict = {"Recording type": recording_type, "data": psv_data}
    #
    #         if data_files.get(recording_identifier):
    #             data_files[recording_identifier][recording_type] = data_dict
    #         else:
    #             data_files[recording_identifier] = {}
    #             data_files[recording_identifier]["Group number"] = list_of_elements[0]
    #             data_files[recording_identifier]["Recording number"] = list_of_elements[1]
    #             data_files[recording_identifier]["Participant ID"] = list_of_elements[2]
    #             data_files[recording_identifier][recording_type] = data_dict
    #             # A list that will get the time of handwashing. It is a list in case there are more than one handwashing events in the recording
    #             data_files[recording_identifier]['Handwashing time'] = []
    #             #the next one will help us to distinguish between protocol recordings to regular one
    #             data_files[recording_identifier]['Protocol'] = 0


    #print(data_files["02_01_B"])

    hand_washing_duration_list  = []
    keys = data_files.keys()
    for key in data_files.keys():
        recording = data_files[key]['label']['data']
        # we get the label table
        if recording['End (Seconds from Recording Start)'.upper()].iloc[-1] < 330:
            data_files[key] ["Protocol"] += 1
        #now let's have only the phases of handwashing. a check is conducted to make sure no miss-labeling
        relevant_phases = recording[(recording["Label".upper()]==1) & (recording['Description'.upper()].isin(['HandWashing', 'Hand Washing','Handwashing', 'washing', 'Washing']))]
        phases_by_name = recording[(recording['Description'.upper()].isin(['HandWashing', 'Hand Washing','Handwashing', 'washing', 'Washing']))]
        phases_by_number = recording[(recording["Label".upper()]==1)]
        if not ((relevant_phases.equals(phases_by_name)) or (relevant_phases.equals(phases_by_number))):
            raise ValueError(f"invalid recording {data_files[key]['Group number']}_{data_files[key]['Recording number']}_{data_files[key]['Participant ID']} - Label do not match actual action")
        
        for index, row in relevant_phases.iterrows():
            # we extract the times of handwashing activities, and also see the Mean and Std to effectively declare windows
            start_col = 'Start (Seconds from Recording Start)'.upper()
            end_col = 'End (Seconds from Recording Start)'.upper()
            start = float(row[start_col])
            end = float(row[end_col])

            total_duration = end - start

            hand_washing_duration_list.append(total_duration)
            data_files[key]['Handwashing time'].append((start, end))


    #print(len(hand_washing_duration_list))
    print(f"the average hand washing duration is: {np.mean(hand_washing_duration_list):3.2f}")
    print(f"the standard deviation of hand washing duration is: {np.std(hand_washing_duration_list):3.2f}")

    #לשנות את זה בהתאם למה שדביר אומר

    def create_window (window_size, window_step, recording):
        #at first, we commit check that each recording starts with 0. if not, we "normalize" the data by creating a reset to zero
        for sensor in ["Acc","Gyro","Mag"]:
            starting_point = recording[sensor]['data']['elapsed (s)'.upper()].min()
            if starting_point !=0:
                for index in time_column.index.tolist():
                    recording[sensor]['data']['elapsed (s)'.upper()][index] -= starting_point
                    for time_tuple in recording['Handwashing time']:
                        #if the data table also do not start from zero.
                        time_tuple[0] -= starting_point
                        time_tuple[1] -= starting_point
            time_column = recording[sensor]['data']['elapsed (s)'.upper()]
            recording_start = time_column.min()
            recording_end = time_column.max()

            window_starting_time = 0

        window_starting_points = []
        window_ending_points = []


        while window_starting_time + window_step  <= recording_end:
            window_starting_points.append(window_starting_time)
            window_ending_time = window_starting_time + window_size
            if window_ending_time > recording_end:
                #in case we pass the limit of the end of recording with the new end of window,
                # the end point will be the actual ending of the recording
                #להתייעץ על זה
                window_ending_points.append(recording_end)
                break
            window_ending_points.append(window_ending_time)
            window_starting_time += window_step

        return window_starting_points, window_ending_points

    def assign_label_to_window(window_starting_time, window_ending_time,referance_timing_list):
            #להתייעץ על זה
            # the intention of this function is to assign the window into a certain label.
            # as rule of decision, we decided that if the majority of the window (meaning 50% or above) is Handwashing, it will be considered as handwashing. other wise, it will not
            for timing in referance_timing_list:
                if timing[0] <= window_starting_time <= timing[1] and timing[0] <= window_ending_time <= timing[1]:
                    #in this case the entire windows is between the handwashing edgepoints
                    return 1
                elif timing[0] <= window_starting_time<= timing[1] and  timing[1]<= window_ending_time:
                    #in this case the windows starts between the handwashing edgepoints but ends afterwards.
                    # So if more than 50% of it is handwashing, the label will be 1. otherwise, 0.
                    if timing[1] - window_starting_time >= window_ending_time - timing[1]:
                        return 1
                    else:
                        continue
                elif window_starting_time <= timing[0] and timing[0] <= window_ending_time <= timing[1]:
                    #in this case the windows starts before the handwashing events but ends between the handwashing edgepoints.
                    # So if more than 50% of it is handwashing, the label will be 1. otherwise, 0.
                    if timing[0] - window_starting_time >= window_ending_time - timing[0]:
                        return 1
                elif  window_starting_time <= timing[0] and timing[1]<= window_ending_time:
                    #in this case, the handwashing is shorter than the window and included inside it.
                    #so if it is more than 50% of the window we consider it as handwashing.
                    if window_ending_time - window_starting_time <= 2 * (timing[1] - timing[0]):
                        return 1
                    else:
                    # i get continue in any case of mismatch because there can be more than one event.
                    #after all, i get zero.
                        continue
                elif window_ending_time <= timing[0] or timing[1]<= window_starting_time:
                    continue
            return 0
    columns_names = ['First second of the activity', 'Last second of the activity', 'Participant ID','Group number','Recording number', 'Protocol', 'Label']
    X_matrix = pd.DataFrame([], columns = columns_names)

    #We now go over each recording and creates the windows, and labeling each one of them
    for recording in data_files.values():
            window_starting_points, window_ending_points = create_window(window_size, window_step, recording)
            for i in range(len(window_starting_points)): #We go over each window
                window_starting_time = window_starting_points[i]
                window_ending_time = window_ending_points[i]
                label = assign_label_to_window(window_starting_time = window_starting_time, window_ending_time = window_ending_time, referance_timing_list = recording['Handwashing time'])

                #Now, creating the dict that will be added to the DF
                row_dict = {'First second of the activity': window_starting_time, 'Last second of the activity': window_ending_time, 'Participant ID': recording['Participant ID'], 'Group number':  recording['Group number'], 'Recording number': recording['Recording number'], 'Protocol':  recording['Protocol'], 'Label': label}
                X_matrix.loc[len(X_matrix.index)] = row_dict
                print(f"row {len(X_matrix.index)-1} has been added")

    Y_vector = X_matrix['Label']
    X_matrix = X_matrix.drop(columns=['Label'])



    #print(X_matrix[(X_matrix["First second of the activity"] != 0) &(X_matrix["Last second of the activity"] != 0)][:1000])
    #print(Y_vector)

    return X_matrix, Y_vector



data_path = r"C:\Users\nirei\OneDrive\Desktop\Bachelors Degree - Biomedical Engineering And Neuroscience\Year 4\Semester A\Continuous Monitoring of Physiological Parameters\PythonProject7\02"
#We call the function 3 times to get 3 window sizes
X_matrix_1, Y_vector_1 = segment_signal(data_path, 3, 1.5)
#X_matrix_2, Y_vector_2 = segment_signal(data_path, 10, 3)
#X_matrix_3, Y_vector_3 = segment_signal(data_path, 22, 4)

#def combine_matrices (X_matrix_1, Y_vector_1, X_matrix_2, Y_vector_2, X_matrix_3, Y_vector_3):
    #  #We add the matrices as concatination of table as they have the same column but just different window times
    #combined_x_matrix = pd.concat([X_matrix_1, X_matrix_2, X_matrix_3], axis=0)
    #combined_y_vector = pd.concat([Y_vector_1, Y_vector_2, Y_vector_3], axis=0)
    #combined_x_matrix = combined_x_matrix.reset_index(drop=True)
    #combined_y_vector = combined_y_vector.reset_index(drop=True)
    #return combined_x_matrix, combined_y_vector


#X_matrix, Y_vector =  combine_matrices (X_matrix_1, Y_vector_1, X_matrix_2, Y_vector_2, X_matrix_3, Y_vector_3)



#print(X_matrix['Acc_X-AXIS'])
#print(X_matrix['Mag_Y-AXIS'])

#לזכור להוסיף מעין תרגום של הזמן של החלון לזמן של הדוגם בפועל

##-------Part B: Feature Extraction-------

#baseline wander
#לחשוב על הevent trigger

def extract_features (data_path, X_matrix):
    data_files = load_data(data_path)
    num_features = 0


    def fix_baseline_wander (data, sampling_frequency, filter_order = 5, cutoff_frequency = 0.5):
        #The idea is to compute baseline wander in order to create more representative view of the data.
        #In order to do so, Butterworth HPF is applied on the data, in order to get rid of the low frequencies's noise.
        #The sampling frequency is known as it was defined by us, and the cutoff_frequency was chosen to be 0.5 as most physiological movement are at 1 Hz and more.
        nyquist_frequency = sampling_frequency * 0.5
        normal_cutoff = cutoff_frequency / nyquist_frequency
        b, a = butter(filter_order, normal_cutoff, btype='high', analog=False)
        corrected_data = filtfilt(b, a, data)

        return corrected_data


    def normalize_data(data_values, method = 'IQR'):

        if data_values.ndim == 1:
            #to use the sk function, we need to transform the shape
            data_values = data_values.reshape(-1, 1)

        if method == 'IQR':
            """
            This is normalization in the robust way, which is less manipulated by outliers.
            the IQR method allows choosing normalization the do not depend on the radical values
            so we decided to go for normalization between the 1% percentile and 99% percentile,
            considering the highest 1% lowest and highest values to be outliers.
            it also takes the median that is less dependant on extreme values but rater on the entire data distribution.
            normalization_meth = sk.preprocessing.RobustScaler(quantile_range=(1.0, 99.0))
            """

        elif method == 'standard':
            normalization_meth = sk.preprocessing.StandardScaler()

        elif method == 'MinMax':
            normalization_meth = sk.preprocessing.MinMaxScaler()

        normalized_data = normalization_meth.fit_transform(data_values)

        return normalized_data. flatten()

    columns_names = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS',
                     'Mag_X-AXIS', 'Mag_Y-AXIS', 'Mag_Z-AXIS', 'Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS',
                     'Mag_X-AXIS', 'Mag_Y-AXIS', 'Mag_Z-AXIS']
    for col in columns_names:
        X_matrix[col] = pd.Series(dtype=object)

    def find_closest_neighbor(points_series, fixed_point):

        ##This function takes points_series which is the actual time points of the sensor,
        ## and fixed_point which is the edge point of the window, and tries to match them by finding
        ## the closet point to the edge point of the window in the data.

        diff = (points_series - fixed_point).abs()
        index = diff.idxmin()
        return index


    def applying_windows(recording, X_matrix, ):
        #This is a function that meant to save the actual data of each window.
        #it gets as an input a certain recording dict and x matrix, and iterate over the entire windows of this recording,
        #in order to save the data for every sensor in each axis.
        #then, in the row with the time points of the window, it saves the actual data of each window.
        ID = recording['Participant ID']
        group_number = recording['Group number']
        recording_number = recording['Recording number']

        # Filter the entire X_matrix to get the relevant rows for the current recording
        time_points_data = X_matrix[(X_matrix['Participant ID'] == ID) & (X_matrix['Group number'] == group_number) & ( X_matrix['Recording number'] == recording_number)]
        for sensor_name in ["Acc", "Gyro", "Mag"]:
            if sensor_name == "Acc":
                unit =" (G)"
            elif sensor_name == "Mag":
                unit =" (T)"
            elif sensor_name == "Gyro":
                unit =" (deg/s)".upper()
            sensor_data = recording[sensor_name]['data']
            time_series = sensor_data['elapsed (s)'.upper()]

            #Iterate over theindex label of the filtered DataFrame, with the row data
            for label, row in time_points_data.iterrows():
                starting_point = row['First second of the activity']
                ending_point = row['Last second of the activity']

                # Find the indices in the sensor data
                adjusted_starting_index = find_closest_neighbor(time_series, starting_point)
                adjusted_ending_index = find_closest_neighbor(time_series, ending_point)

                for ax in ("X-AXIS", "Y-AXIS", "Z-AXIS"):
                    #axis_data = sensor_data[[c for c in sensor_data.columns if ax.split('-')[0].lower() in c.lower()]].values.ravel()
                    axis_data = sensor_data[ax + unit].values
                    window_data = np.array(axis_data[adjusted_starting_index:adjusted_ending_index])
                    column_name = sensor_name + '_' + ax
                    X_matrix.at[label, column_name] = [window_data]

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
                if (recording['Participant ID'] == 'A') & (recording['Group number'] == '02') & (recording['Recording number'] == '01'):
                    plt.figure()
                    plt.subplot(1,2,1)
                    plt.title("before baseline wander")
                    plt.plot(recording[sensor_name]['data']['elapsed (s)'.upper()],axis_data)
                    plt.subplot(1,2,2)
                    plt.title("after baseline wander")
                    plt.plot(recording[sensor_name]['data']['elapsed (s)'.upper()],new_axis_data)
                    plt.show()
                # Conducting normalization on the entire data from a certain recording in a certain sensor - only if we want normaliation
                #if normalization == True:
                    #new_axis_data = normalize_data(new_axis_data)
                recording[sensor_name]['data'][axis + unit] = new_axis_data
        applying_windows(recording, X_matrix)

    def compute_signal_magnitude(sensor_data):
        #We found that metrics preformed on signal magnitude can be a very good predictor in such things.
        #We want to calculate the magnitude of all 3 axis.
        #We sum their squares and then put it in a square
        sensor_data_x = np.array(sensor_data.iloc[0]).ravel()
        sensor_data_y = np.array(sensor_data.iloc[1]).ravel()
        sensor_data_z = np.array(sensor_data.iloc[2]).ravel()
        Nx = len(sensor_data_x)
        Ny = len(sensor_data_y)
        Nz = len(sensor_data_y)

        N = min([Nx, Ny, Nz])  # We want to have a comparison and for that we will neglect a mistake of 1-2 non- aligned time points by cutting them
        sensor_data_x = sensor_data_x[:N]
        sensor_data_y = sensor_data_y[:N]
        sensor_data_z = sensor_data_z[:N]

        # we check if one of them is NaN
        is_x_nan = np.isnan(sensor_data_x).all()
        is_y_nan = np.isnan(sensor_data_y).all()
        is_z_nan = np.isnan(sensor_data_z).all()

        if is_x_nan or is_y_nan or is_z_nan:
                return np.nan
        sensor_data_x = np.array(sensor_data_x)
        sensor_data_y = np.array(sensor_data_y)
        # As the sampling frequency of z is two time lower, we get half the points for z
        # in comparison to the points of x and y. To fix that, we multiply by a factor of 2 the number of points of z
        # and then assign the unreal points with the value of the points before.
        # not ideal, but necessary for this metric.


        return np.sqrt(sensor_data_x ** 2 + sensor_data_y ** 2 + sensor_data_z ** 2)

    X_features = copy.deepcopy(X_matrix) #creating X feature before feature extraction

    for sensor_name in ["Acc", "Gyro", "Mag"]:
        X_features[sensor_name + "_SM"] = X_features[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(compute_signal_magnitude,axis=1)
        columns_names.append(sensor_name + "_SM")
        num_features += 1


    def safe_unwrap(data_list):
        # this function is meant to assure that we deal with a data which is not NaN or empty.
        # it first check whether it is a list or pd.Series and then transforms it into an np array, and if it is already np array - it does anything.
        # At the end, if the size is x it is full of NaN - it returns None
        if isinstance(data_list, (list, pd.Series)):
                x = np.array(data_list).ravel()
        else:
                x = data_list

        if x.size == 0 or np.isnan(x).all():
                return None
        return x

        # הסבר: קודם כל ולפני הכל - אני אעשה את זה באופן כללי, ואז להריץ לולאה לכל הפיצ'ר למעט אלו הכוללים. לאחר מכן, לודא שעושהה את זה שונה עבור שלושת הצירים יחד

    def add_frequency_domain_features(df, column_list, num_features):
        # This function is meant to find features dependent on the frequency domain.
        # it receives as input a dataframe to conduct the calculation on and to add the features to,
        # a list of columns to calculate based on them, and at last - num_features that will be used for tracking the number of features addded.

        def PSD_for_signal(signal,sampling_rate=50):
            from scipy.signal import welch
            #This function receives a signal and calculate its PSD and frequencies.
            #PSD is the measure of signal's power content versus frequency, and it is more useful for cases where the signal is noisy and not harmonic.
            frequencies, psd = welch(signal, fs=sampling_rate, nperseg=len(signal))
            #nperseg decides how many samples will be calculated in each iteration of PSD analyzing.
            #we take the number of samples in the window.
            #we get as output the vector of frequencies and the vector of PSD values that fit the matching frequency
            return frequencies, psd


        def calculate_frequency_domain_features(data_list,sampling_rate=50):
            from scipy.stats import entropy

            data_list = safe_unwrap(data_list)
            if data_list is not None:
                frequencies, psd = PSD_for_signal(data_list, sampling_rate)

                # let's find spectral entropy.
                # Measures how much energy is spread across frequencies (high entropy) or concentrated (low entropy).
                norm_psd = psd / np.max(psd)
                spectral_entropy = entropy(norm_psd)

                # let's find the total Energy of the signal in the frequency domain
                # It may indicate about the how powerfull is the signal.
                total_energy = np.sum(psd)

                # let's find the Frequency Centroid.
                # It is the mass center of the spectrum,
                # and its value (high / low) will show about the distribution of the signal (more dominant on the high/low frequencies)
                freq_centroid = np.sum(frequencies * psd) / np.sum(psd)

                # let's find the Band Dominant Frequency.
                # This is the frequency in which the PSD is the highest.
                # will help identify harmonic movement
                dominant_freq = frequencies[np.argmax(psd)]

                # let's find the frequency variance, which show wide is the spectrum
                freq_variance = np.sum(((frequencies - freq_centroid) ** 2) * psd) / np.sum(psd)

            return spectral_entropy,total_energy, freq_centroid, dominant_freq, freq_variance

        new_columns = {}
        feature_suffixes = ['spectral_entropy', 'total_energy', 'frequency_centroid',
                            'dominant_frequency', 'frequency_variance']
        for column in column_list:
            #For each sensor, we will calculate each metric for every axis, and also for the magnitude of all the axes combined
            sampling_frequency = 50
            if 'Mag' in column:
                sampling_frequency = 25
            features_series = df[column].apply(
                lambda x: pd.Series(
                    calculate_frequency_domain_features(x, sampling_frequency),
                    index=feature_suffixes  #  defining the name of returned columns
                )
            )
            #adding to the dict
            for suffix in feature_suffixes:
                col_name = f"{column}_{suffix}"
                new_columns[col_name] = features_series[suffix]
                print(f"added {col_name} column")

            num_features += len(feature_suffixes)
        df_new = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        return df_new, num_features

    X_features, num_features = add_frequency_domain_features(X_features, columns_names, num_features)

    #At this point, the frequency domain features will be extracted.
    #The reason for that, is that normalization may change frequency domain behaviour, as it includes scaling.
    #Thus, normalization will be conducted later

    for recording in data_files.values():
        #adding the window's data for the suitable row
        for sensor_name in ["Acc", "Gyro", "Mag"]:
            if sensor_name == "Acc":
                unit =" (G)"
            elif sensor_name == "Mag":
                unit =" (T)"
            elif sensor_name == "Gyro":
                unit =" (deg/s)".upper()
            for axis in ["X-AXIS", "Y-AXIS", "Z-AXIS"]:
                axis_data = recording[sensor_name]['data'][axis + unit].values
                # Conducting normalization on the entire data from a certain recording in a certain sensor
                normalized_axis_data = normalize_data(axis_data)
                recording[sensor_name]['data'][axis + unit] = normalized_axis_data
        applying_windows(recording, X_features)

    #Now, let's find the magnitude again, this time normalized.
    for sensor_name in ["Acc", "Gyro", "Mag"]:
        X_features[sensor_name + "_SM"] = X_features[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(compute_signal_magnitude,axis=1)
        columns_names.append(sensor_name + "_SM")
        num_features += 1

    def add_basic_metrics(df, column_names, num_features):
        #This method

        def calculate_list_mean(data_list):
               #we try to get the mean of the data of a cell
               data_list = safe_unwrap(data_list) # making the check about the data list
               if data_list is not None:
                   return np.mean(data_list)
               else:
                    return np.nan

        def calculate_list_median(data_list):
               #we try to get the median of the data of a cell
                data_list = safe_unwrap(data_list) #making the check about the data list
                if data_list is not None:
                    return np.median(data_list)
                else:
                    return np.nan

        def calculate_list_STD(data_list):
               #we try to get the std of the data of a cell
               data_list = safe_unwrap(data_list)  # making the check about the data list
               if data_list is not None:
                   return np.std(data_list)
               else:
                   return np.nan


        def calculate_list_power(data_list):
               #we try to get the energy of the data of a cell. we transform it to a numpy array and then calculated the square of the sum of its particles,
               #and then we divide it by N in order to prevent bias caused because of larger windows.
                data_list = safe_unwrap(data_list)  # making the check about the data list
                if data_list is not None:
                    N = len(data_list)
                    x = np.array(data_list)
                    x_squared = x**2
                    power = np.sum(x_squared) / N
                    return power

                else:
                    return np.nan

        def calculate_list_IQR(data_list):
                #we calculate the IQR of the data by getting the 25th and 75th percentile and calculating the difference between them
                data_list = safe_unwrap(data_list)  # making the check about the data list
                if data_list is not None:
                    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
                    Q1 = np.percentile(data_list, 25)
                    Q3 = np.percentile(data_list, 75)

                    # Calculate IQR
                    IQR = Q3 - Q1
                    return IQR
                else:
                    return np.nan

        def calculate_list_max(data_list):
                #we calculate the maximum value of the data
                data_list = safe_unwrap(data_list)  # making the check about the data list
                if data_list is not None:
                    max_value = np.max(data_list)
                    return max_value
                else:
                    return np.nan

        def calculate_list_min(data_list):
                # we calculate the minimum value of the data
                data_list = safe_unwrap(data_list)  # making the check about the data list
                if data_list is not None:
                    max_value = np.min(data_list)
                    return max_value
                else:
                    return np.nan


        def calculate_peak_to_peak_difference (data_list):
                data_list = safe_unwrap(data_list)
                if data_list is not None:
                    max = calculate_list_max(data_list)
                    min = calculate_list_min(data_list)
                    #max_index = np.argmax(data_list)
                    peak_to_peak_difference = max - min
                    return peak_to_peak_difference
                else:
                    return np.nan

        def calculate_zero_crossing (data_list):
                data_list = safe_unwrap(data_list)  # making the check about the data list
                if data_list is not None:
                    number_of_zero_crossings = np.sum(np.diff(np.sign(data_list)) !=0) #applying mask of boolean condition on difference between the sign valuse of adjacent points
                    return number_of_zero_crossings
                else:
                    return np.nan





        def calculate_list_RMS(data_list):
                #this method is meant to find the RMS of the data
                data_list = safe_unwrap(data_list)  # making the check about the data list
                if data_list is not None:
                    power = calculate_list_power(data_list)
                    RMS = np.sqrt(power)
                    return RMS

                else:
                    return np.nan

        def calculate_sensor_RMS(sensor_data):
                #The function recieves as an input of sensor fata -  3 cells of the axes data
                #We want to calculate the RMS of all 3 axis.
                #we pass the function the data from all three axes and try to get the RMS of the combined array, that take the points from all axes into considerations
                sensor_data_x = np.array(sensor_data.iloc[0]).ravel()
                sensor_data_y = np.array(sensor_data.iloc[1]).ravel()
                sensor_data_z = np.array(sensor_data.iloc[2]).ravel()
                Nx = len(sensor_data_x)
                Ny = len(sensor_data_y)
                Nz = len(sensor_data_z)
                N = min([Nx, Ny, Nz]) #We want to have a comparison and for that we will neglect a mistake of 1-2 non- aligned time points by cutting them
                sensor_data_x = sensor_data_x[:N]
                sensor_data_y = sensor_data_y[:N]
                sensor_data_z = sensor_data_z[:N]

                #we check if one of them is NaN
                is_x_nan = np.isnan(sensor_data_x).all()
                is_y_nan = np.isnan(sensor_data_y).all()
                is_z_nan = np.isnan(sensor_data_z).all()

                if is_x_nan or is_y_nan or is_z_nan:
                    return np.nan


                RMS = np.sqrt(np.sum(sensor_data_x ** 2 + sensor_data_y ** 2 + sensor_data_z ** 2)/N)
                return RMS

            #in order to achieve better running time, we will add a dict containing all the new columns, and then add them together
        new_columns = {}

        for column in column_names:
                new_columns[column + '_mean'] = df[column].apply(calculate_list_mean)
                print(f"added {column + '_mean'} column")
                print(new_columns[column + '_mean'])
                num_features += 1
                new_columns[column + '_std'] = df[column].apply(calculate_list_STD)
                print(f"added {column + '_std'} column")
                print(new_columns[column + '_std'])
                num_features += 1
                new_columns[column + '_median'] = df[column].apply(calculate_list_median)
                print(f"added {column + '_median'} column")
                print(new_columns[column + '_median'])
                num_features += 1
                new_columns[column + '_RMS'] = df[column].apply(calculate_list_RMS)
                print(f"added {column + '_RMS'} column")
                print(new_columns[column + '_RMS'])
                num_features += 1

                new_columns[column + '_IQR'] = df[column].apply(calculate_list_IQR)
                print(f"added {column + '_IQR'} column")
                print(new_columns[column + '_IQR'])
                num_features += 1

                new_columns[column + '_max'] = df[column].apply(calculate_list_max)
                print(f"added {column + '_max'} column")
                print(new_columns[column + '_max'])
                num_features += 1

                new_columns[column + '_min'] = df[column].apply(calculate_list_min)
                print(f"added {column + '_min'} column")
                print(new_columns[column + '_min'])
                num_features += 1

                new_columns[column + '_peak_to_peak'] = df[column].apply(calculate_peak_to_peak_difference)
                print(f"added {column + '_peak_to_peak'} column")
                print(new_columns[column + '_peak_to_peak'])
                num_features += 1

                new_columns[column + '_number_of_zero_crossing'] = df[column].apply(calculate_zero_crossing)
                print(f"added {column + '_number_of_zero_crossing'} column")
                print(new_columns[column + '_number_of_zero_crossing'])
                num_features += 1

                # df[column + '_power'] = df[column].apply(calculate_list_power) # it will be used for finding power afterwards, and then be deleted
                # print("added power column")

            #let's calculate the RMS of every axis for every sensor
        for sensor_name in ["Acc", "Gyro", "Mag"]:
                    new_columns[sensor_name + '_RMS_Total'] = df[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(calculate_sensor_RMS,axis=1)
                    print(f"added {sensor_name + '_RMS_Total'} column")
                    print(new_columns[sensor_name + '_RMS_Total'])
                    num_features += 1

        #Now we concatenate the newly created dict to the df - just one addition
        df_new = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        return df_new,num_features

    X_features,num_features = add_basic_metrics(X_features, columns_names, num_features)

    def add_disribution_features(df, column_list, num_features):
        #This function is meant to find features which are connected to the distribution.
        #it receives as input a dataframe to conduct the calculation on and to add the features to,
        #a list of columns to calculate based on them, and at last - num_features that will be used for tracking the number of features addded.
        def compute_skewness(data_list):
            from scipy.stats import skew
            # Skewness is a statistical metric to find whether the distribution of the data is symmetric or asymmetric.
            # If it is indeed asymmetric - skewness try to measure whether the asymmetricity is towards the positive end or the negative end, meaning if it is towards the higher of lower values.
            # This method may supply key insights on the distribution of the data and for that it can be used in the model, as it basically indicates the shape and size of variation on either side of the central value.
            # It also indicates how far the distribution differs from the normal distribution.
            data_list = safe_unwrap(data_list)
            if data_list is not None:
                skewness = skew(data_list)
                return skewness
            else:
                return np.nan

        def compute_kurtosis(data_list):
            from scipy.stats import kurtosis
            # While skewness focuses on the spread (tails) of normal distribution, kurtosis focuses more on the height.
            # It tells us how peaked or flat our normal (or normal-like) distribution is
            #High kurtosis indicates heavier tales and more sharp peaks, while low kurtosis means flat peaks and lighter tails.
            # It may include data regrading the behaviour of the data, and by that may be useful.
            data_list = safe_unwrap(data_list)
            if data_list is not None:
                kurtosis = kurtosis(data_list, fisher = True)
                return kurtosis
            else:
                return np.nan

        # in order to achieve better running time, we will add a dict containing all the new columns, and then add them together
        new_columns = {}
        for column in column_list:
            new_columns[column + '_kurtosis'] = df[column].apply(compute_skewness)
            print(f"added {column + '_kurtosis'} column")
            print(new_columns[column + '_kurtosis'])
            num_features += 1
            new_columns[column + '_skewness'] = df[column].apply(compute_kurtosis)
            print(f"added {column + '_skewness'} column")
            print(new_columns[column + '_skewness'])
            num_features += 1
        #Now we concatenate the newly created dict to the df - just one addition
        df_new = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        return df_new, num_features

    X_features, num_features = add_disribution_features(X_features, columns_names, num_features)

    def time_dependent_features(df, column_list, num_features):

        def calculate_slew_rate(data_list):
            #we calculate the slew rate, which is the maximal change between two adjacent points
            data_list = safe_unwrap(data_list)  # making the check about the data list
            if data_list is not None:
                slew_rate = np.max(np.abs(np.diff(data_list)))
                return slew_rate
            else:
                return np.nan

        #def calculate_area_under_absolute_values(data_list):

        # def calculate_rise_time(data_list):
        #     #we calculate the rise time, which is the time took the signal get from the 10% percentile value to 90% percentile value
        #     data_list = safe_unwrap(data_list)  # making the check about the data list
        #     if data_list is not None:
        #         point_of_10 = np.percentile(data_list, 0.1)
        #         point_of_90 = np.percentile(data_list, 0.9)
        #         index_of_10 = np.where(data_list == point_of_10)
        #         index_of_90 = np.where(data_list == point_of_90)
        #
        #         rise_time = abs(index_of_90 - index_of_10)
        #
        #         return rise_time
        #     else:
        #         return np.nan
    def EMD_properties(df, column_list, num_features):
        def find_imfs(data_list):
            data_list = safe_unwrap(data_list)  # making the check about the data list
            if data_list is not None:
                imfs = PyEMD.EMD(data_list)
                return imfs
            else:
                return np.nan

        def find_imfs_properties(data_list):
            #It seems like the main factors are the std and relative energy of the imfs, so this will be our focus
            data_list = safe_unwrap(data_list)
            emd = PyEMD.EMD()
            imfs = emd(data_list)
            total_energy = np.sum(imfs ** 2)
            imfs_std = []
            relative_imf_energy = []

            for i in range(2):
              imfs_std.append(np.std(imfs[:,i]))
              relative_imf_energy.append(imfs[:,i]**2 / total_energy)
            return imfs_std[0],imfs_std[1], relative_imf_energy[0], relative_imf_energy[1]

        new_columns = {}
        feature_suffixes = ['imf1 relative energy', 'imf2 relative energy', 'imf1 std',
                            'imf2 std']
        for column in column_list:
            # For each sensor, we will calculate each metric for every axis, and also for the magnitude of all the axes combined
            features_series = df[column].apply(find_imfs_properties,index=feature_suffixes)
            # adding to the dict
            for suffix in feature_suffixes:
                col_name = f"{column}_{suffix}"
                new_columns[col_name] = features_series[suffix]
                print(f"added {col_name} column")

            num_features += len(feature_suffixes)
        df_new = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        return df_new, num_features

    #adding the imf traits
    X_features, num_features = EMD_properties(X_features, columns_names, num_features)
    #getting rid of the columns with the vectors of values
    X_features.drop(labels = columns_names, axis = 1)

    print(f"added {num_features} columns")

    return X_features

X_features = extract_features (data_path, X_matrix_1)

X_features.to_excel("X_features.xslx", index=False)

##-------Part C: Train & Test -------

##-------Part D: Feature Correlation Analysis -------



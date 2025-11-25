import copy

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
                window_ending_points.append(window_ending_time)
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
X_matrix_2, Y_vector_2 = segment_signal(data_path, 10, 3)
X_matrix_3, Y_vector_3 = segment_signal(data_path, 22, 4)

def combine_matrices (X_matrix_1, Y_vector_1, X_matrix_2, Y_vector_2, X_matrix_3, Y_vector_3):
    #We add the matrices as concatination of table as they have the same column but just different window times
    combined_x_matrix = pd.concat([X_matrix_1, X_matrix_2, X_matrix_3], axis=0)
    combined_y_vector = pd.concat([Y_vector_1, Y_vector_2, Y_vector_3], axis=0)
    combined_x_matrix = combined_x_matrix.reset_index(drop=True)
    combined_y_vector = combined_y_vector.reset_index(drop=True)
    return combined_x_matrix, combined_y_vector

X_matrix, Y_vector =  combine_matrices (X_matrix_1, Y_vector_1, X_matrix_2, Y_vector_2, X_matrix_3, Y_vector_3)



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
            #This is normalization in the robust way, which is less manipulated by outliers.
            #the IQR method allows choosing normalization the do not depend on the radical values
            #so we decided to go for normalization between the 1% percentile and 99% percentile,
            #considering the highest 1% lowest and highest values to be outliers.
            #it also takes the median that is less dependant on extreme values but rater on the entire data distribution.
            normalization_meth = sk.preprocessing.RobustScaler(quantile_range=(1.0, 99.0))

        elif method == 'standard':
            normalization_meth = sk.preprocessing.StandardScaler()

        elif method == 'MinMax':
            normalization_meth = sk.preprocessing.MinMaxScaler()

        normalized_data = normalization_meth.fit_transform(data_values)

        return normalized_data. flatten()

    columns_names = ['Acc_X-AXIS', 'Acc_Y-AXIS', 'Acc_Z-AXIS', 'Gyro_X-AXIS', 'Gyro_Y-AXIS', 'Gyro_Z-AXIS',
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


    def applying_windows(recording, X_matrix):
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


                    # Extract the window data


    for recording in data_files.values():
        #adding the window's data for the suitable row
        for sensor_name in ["Acc", "Gyro", "Mag"]:
            sampling_frequency = 50
            if sensor_name == "Acc":
                unit =" (G)"
            elif sensor_name == "Mag":
                unit =" (T)"
                sampling_frequency = 25
            elif sensor_name == "Gyro":
                unit =" (deg/s)".upper()
            for axis in ["X-AXIS", "Y-AXIS", "Z-AXIS"]:
                axis_data = recording[sensor_name]['data'][axis + unit].values
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
                new_axis_data = normalize_data(new_axis_data)
                recording[sensor_name]['data'][axis + unit] = new_axis_data
        applying_windows(recording, X_matrix)


        #X_features = X_matrix.copy.deepcopy() #creating X feature before feature extraction
        X_features = X_matrix #creating X feature before feature extraction

        def compute_signal_magnitude(sensor_data):
            #We found that metrics preformed on signal magnitude can be a very good predictor in such things.
            #We want to calculate the magnitude of all 3 axis.
            #we sum their squares and then put it in a square
            sensor_data_x = np.array(sensor_data.iloc[0]).ravel()
            sensor_data_y = np.array(sensor_data.iloc[1]).ravel()
            sensor_data_z = np.array(sensor_data.iloc[2]).ravel()
            Nx = len(sensor_data_x)
            Ny = len(sensor_data_y)
            Nz = len(sensor_data_y)

            N = min([Nx,
                     Ny, Nz])  # We want to have a comparison and for that we will neglect a mistake of 1-2 non- aligned time points by cutting them
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

        for sensor_name in ["Acc", "Gyro", "Mag"]:
            X_features[sensor_name + "_SM"] = X_features[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(compute_signal_magnitude,axis=1)
            columns_names.append(sensor_name + "_SM")


        def add_basic_metrics(df, column_names):
            #,לשקול להוסיף IQR מקסימום, מינימום, פיק טו פיק ועוד ערכים נוספים

            def calculate_list_mean(data_list):
               #we try to get the mean of the data of a cell
                if isinstance(data_list, list) and data_list:

                    return np.mean(data_list)
                return np.nan

            def calculate_list_median(data_list):
               #we try to get the median of the data of a cell
                if isinstance(data_list, list) and data_list:
                    return np.median(data_list)
                return np.nan

            def calculate_list_STD(data_list):
               #we try to get the std of the data of a cell
                if isinstance(data_list, list) and data_list:
                    return np.std(data_list)
                return np.nan


            def calculate_list_power(data_list):
               #we try to get the energy of the data of a cell. we transform it to a numpy array and then calculated the square of the sum of its particles,
               # and then we divide it by N in order to prevent bias caused because of larger windows.
                if data_list:
                    N = len(data_list)
                    x = np.array(data_list)
                    x_squared = x**2
                    power = np.sum(x_squared) / N
                    return power

                else:
                    return np.nan

                return np.nan


            def calculate_list_RMS(data_list):
                #לחשוב על זה
                if data_list:
                    power = calculate_list_power(data_list)
                    RMS = np.sqrt(power)
                    return RMS

                else:
                    return np.nan

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




            for column in column_names:
                df[column + '_mean'] = df[column].apply(calculate_list_mean)
                print(f"added {column + '_mean'} column")
                print(df[column + '_mean'])
                df[column + '_std'] = df[column].apply(calculate_list_STD)
                print(f"added {column + '_std'} column")
                print(df[column + '_std'])
                df[column + '_median'] = df[column].apply(calculate_list_median)
                print(f"added {column + '_median'} column")
                print(df[column + '_median'])
                df[column + '_RMS'] = df[column].apply(calculate_list_RMS)
                print(f"added {column + '_RMS'} column")
                print(df[column + '_RMS'])
                # df[column + '_power'] = df[column].apply(calculate_list_power) # it will be used for finding power afterwards, and then be deleted
                # print("added power column")

            #let's calculate the RMS of every axis for every sensor
            for sensor_name in ["Acc", "Gyro", "Mag"]:
                    df[sensor_name + '_RMS_Total'] = df[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(calculate_sensor_RMS,axis=1)
                    print(f"added {sensor_name + '_RMS_Total'} column")
                    print(df[sensor_name + '_RMS_Total'])

    add_basic_metrics(X_features, columns_names)









    return X_features

X_features = extract_features (data_path, X_matrix)

print(X_features)
##-------Part C: Train & Test -------

##-------Part D: Feature Correlation Analysis -------



from scipy.signal import butter, filtfilt
import sklearn as sk
import pandas as pd
import numpy as np
import PyEMD
#from joblib import Parallel, delayed

##-------Part B: Feature Extraction - Helper Functions-------

#we will define the methods that calculate the features outside the maing extract_features function for clarity and time-reducing, and then call it in the function

##-------Preprocessing functions-------

def fix_baseline_wander(data, sampling_frequency, filter_order=5, cutoff_frequency=0.5):
    # The idea is to compute baseline wander in order to create more representative view of the data.
    # In order to do so, Butterworth HPF is applied on the data, in order to get rid of the low frequencies's noise.
    # The sampling frequency is known as it was defined by us, and the cutoff_frequency was chosen to be 0.5 as most physiological movement are at 1 Hz and more.
    nyquist_frequency = sampling_frequency * 0.5
    normal_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(filter_order, normal_cutoff, btype='high', analog=False)
    corrected_data = filtfilt(b, a, data)

    return corrected_data


def normalize_data(data_values, method='IQR'):
    if data_values.ndim == 1:
        # to use the sk function, we need to transform the shape
        data_values = data_values.reshape(-1, 1)

    if method == 'IQR':
        # This is normalization in the robust way, which is less manipulated by outliers.
        # the IQR method allows choosing normalization the do not depend on the radical values
        # so we decided to go for normalization between the 1% percentile and 99% percentile,
        # considering the highest 1% lowest and highest values to be outliers.
        # it also takes the median that is less dependant on extreme values but rater on the entire data distribution.

        normalization_meth = sk.preprocessing.RobustScaler(quantile_range=(1.0, 99.0))


    elif method == 'standard':
        normalization_meth = sk.preprocessing.StandardScaler()

    elif method == 'MinMax':
        normalization_meth = sk.preprocessing.MinMaxScaler()

    normalized_data = normalization_meth.fit_transform(data_values)

    return normalized_data.flatten()

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
                axis_data = sensor_data[ax + unit].values
                window_data = np.array(axis_data[adjusted_starting_index:adjusted_ending_index])
                column_name = sensor_name + '_' + ax
                X_matrix.at[label, column_name] = [window_data]

def compute_signal_magnitude(sensor_data):
    # We found that metrics preformed on signal magnitude can be a very good predictor in such things.
    # We want to calculate the magnitude of all 3 axis.
    # We sum their squares and then put it in a square
    sensor_data_x = np.array(sensor_data.iloc[0]).ravel()
    sensor_data_y = np.array(sensor_data.iloc[1]).ravel()
    sensor_data_z = np.array(sensor_data.iloc[2]).ravel()
    Nx = len(sensor_data_x)
    Ny = len(sensor_data_y)
    Nz = len(sensor_data_y)

    N = min([Nx, Ny,
             Nz])  # We want to have a comparison and for that we will neglect a mistake of 1-2 non- aligned time points by cutting them
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

##-------Time-domain features-------##

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


def calculate_list_MAD(data_list):
    # As the handwashing event is relatively rare, the STD may not be ideal.
    # Thus we use MAD- Median Absolute Deviation
    # it is calculated as the median of the diff between the absolute value and the median
    data_list = safe_unwrap(data_list)  # making the check about the data list
    if data_list is not None:
        median_value = np.median(np.mean(data_list))
        abs_values = np.abs(median_value)
        return np.median(abs_values - median_value)
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

def calculate_mean_distance_between_axes(sensor_data):
    # we calculate the mean distance between axes
    # a feature meant for calculating the variance in direction
    sensor_data_x = np.array(sensor_data.iloc[0]).ravel()
    sensor_data_y = np.array(sensor_data.iloc[1]).ravel()
    sensor_data_z = np.array(sensor_data.iloc[2]).ravel()
    Nx = len(sensor_data_x)
    Ny = len(sensor_data_y)
    Nz = len(sensor_data_z)
    N = min([Nx, Ny,
             Nz])  # We want to have a comparison and for that we will neglect a mistake of 1-2 non- aligned time points by cutting them
    sensor_data_x = sensor_data_x[:N]
    sensor_data_y = sensor_data_y[:N]
    sensor_data_z = sensor_data_z[:N]

    # we check if one of them is NaN
    is_x_nan = np.isnan(sensor_data_x).all()
    is_y_nan = np.isnan(sensor_data_y).all()
    is_z_nan = np.isnan(sensor_data_z).all()

    if is_x_nan or is_y_nan or is_z_nan:
        return np.nan

    #let's check for the mean distance between the axes
    x_y_dist = np.abs(sensor_data_x - sensor_data_y)
    z_y_dist = np.abs(sensor_data_z - sensor_data_y)
    x_z_dist = np.abs(sensor_data_x - sensor_data_z)
    mean_dist = np.mean((x_y_dist+ z_y_dist+ x_z_dist) / 3)
    return mean_dist
def calculate_correlation_between_axes(sensor_data):
    # we calculate the correlation coefficients between axes
    # we also look at the mean correlation as the relation between it and the other metric may hint hidden relationship
    sensor_data_x = np.array(sensor_data.iloc[0]).ravel()
    sensor_data_y = np.array(sensor_data.iloc[1]).ravel()
    sensor_data_z = np.array(sensor_data.iloc[2]).ravel()
    Nx = len(sensor_data_x)
    Ny = len(sensor_data_y)
    Nz = len(sensor_data_z)
    N = min([Nx, Ny,
             Nz])  # We want to have a comparison and for that we will neglect a mistake of 1-2 non- aligned time points by cutting them
    sensor_data_x = sensor_data_x[:N]
    sensor_data_y = sensor_data_y[:N]
    sensor_data_z = sensor_data_z[:N]

    # we check if one of them is NaN
    is_x_nan = np.isnan(sensor_data_x).all()
    is_y_nan = np.isnan(sensor_data_y).all()
    is_z_nan = np.isnan(sensor_data_z).all()

    if is_x_nan or is_y_nan or is_z_nan:
        return [np.nan, np.nan, np.nan, np.nan]

    # we find the correlation coefficient of each two axes and the mean correlation.
    # we get the correlation matrix but we look for the correlation coefficient which is in the [0,1]
    if np.std(sensor_data_x) ==0 or np.std(sensor_data_y) == 0:
        x_y_corr = 0
    else:
        x_y_corr = np.corrcoef(sensor_data_x, sensor_data_y)[0, 1]
    if np.std(sensor_data_y) ==0 or np.std(sensor_data_z) == 0:
        y_z_corr = 0
    else:
        y_z_corr =np.corrcoef(sensor_data_z, sensor_data_y)[0, 1]
    if np.std(sensor_data_z) ==0 or np.std(sensor_data_x) == 0:
        x_z_corr = 0
    else:
        x_z_corr =np.corrcoef(sensor_data_x, sensor_data_z)[0, 1]
    mean_corr = np.mean([y_z_corr, x_z_corr, x_y_corr])
    return [x_y_corr, y_z_corr, x_z_corr, mean_corr]


def add_basic_metrics(df, column_names, num_features):
    #This method meant to find basic time dependent metrics to evaluate the data, in align with the functions
 #in order to achieve better running time, we will add a dict containing all the new columns, and then add them together
    new_columns = {}

    for column in column_names:
            new_columns[column + '_mean'] = df[column].apply(calculate_list_mean)
            print(f"added {column + '_mean'} column")
            num_features += 1

            new_columns[column + '_std'] = df[column].apply(calculate_list_STD)
            print(f"added {column + '_std'} column")

            num_features += 1
            new_columns[column + '_median'] = df[column].apply(calculate_list_median)
            print(f"added {column + '_median'} column")

            num_features += 1
            if not "SM" in column:
                new_columns[column + '_RMS'] = df[column].apply(calculate_list_RMS)
                print(f"added {column + '_RMS'} column")
                num_features += 1

            new_columns[column + '_IQR'] = df[column].apply(calculate_list_IQR)
            print(f"added {column + '_IQR'} column")
            num_features += 1

            new_columns[column + '_max'] = df[column].apply(calculate_list_max)
            print(f"added {column + '_max'} column")
            num_features += 1

            new_columns[column + '_min'] = df[column].apply(calculate_list_min)
            print(f"added {column + '_min'} column")
            num_features += 1

            new_columns[column + '_peak_to_peak'] = df[column].apply(calculate_peak_to_peak_difference)
            print(f"added {column + '_peak_to_peak'} column")
            num_features += 1

            new_columns[column + '_number_of_zero_crossing'] = df[column].apply(calculate_zero_crossing)
            print(f"added {column + '_number_of_zero_crossing'} column")
            num_features += 1

            new_columns[column + '_MAD'] = df[column].apply(calculate_list_MAD)
            print(f"added {column + '_MAD'} column")
            num_features += 1

            #this column is calculated for later extracting the dominant power column, and will be soon deleted
            df[column + '_power'] = df[column].apply(calculate_list_power) # it will be used for finding power afterwards, and then be deleted


        #let's calculate the RMS of every axis for every sensor
    for sensor_name in ["Acc", "Gyro", "Mag"]:

                #computing RMS
                new_columns[sensor_name + '_RMS_Total'] = df[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(calculate_sensor_RMS,axis=1)
                print(f"added {sensor_name + '_RMS_Total'} column")

                num_features += 1
                # computing mean distance between axes
                new_columns[sensor_name + '_mean_dist_between_axes'] = df[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(calculate_mean_distance_between_axes,axis=1)
                print(f"added {sensor_name + '_mean_dist_between_axes'} column")

                num_features += 1
                # we also add a column which have the dominant axis energy
                # computing dominant axis energy
                power_columns = df[[sensor_name + '_' + "X-AXIS"+ '_power', sensor_name + '_' + "Y-AXIS"+ '_power', sensor_name + '_' + "Z-AXIS"+ '_power']]
                new_columns[sensor_name + '_dominant_axis_energy'] = power_columns.max(axis=1)
                num_features += 1
                df = df.drop(columns=[sensor_name + '_' + "X-AXIS"+ '_power', sensor_name + '_' + "Y-AXIS"+ '_power', sensor_name + '_' + "Z-AXIS"+ '_power'])

                #computing correlation coefficient between axes
                feature_suffixes = ['X_Y_CORR', 'Z_Y_CORR', 'X_Z_CORR', 'MEAN_AXES_CORR']
                features_series = df[[sensor_name + '_' +"X-AXIS", sensor_name + '_' +"Y-AXIS", sensor_name + '_' + "Z-AXIS"]].apply(
                    lambda x: pd.Series(
                        calculate_correlation_between_axes(x),
                        index=feature_suffixes  #  defining the name of returned columns
                    )
                )
                #adding to the dict
                for suffix in feature_suffixes:
                    col_name = f"{sensor_name}_{suffix}"
                    new_columns[col_name] = features_series[suffix]
                    print(f"added {col_name} column")
                    num_features += 1

                #Now we concatenate the newly created dict to the df - just one addition
    df_new = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df_new,num_features

#Now we move to more advanced time-dependent features

def calculate_slew_rate(data_list):
    #we calculate the slew rate, which is the maximal change between two adjacent points
    data_list = safe_unwrap(data_list)  # making the check about the data list
    if data_list is not None:
        slew_rate = np.max(np.abs(np.diff(data_list)))
        return slew_rate
    else:
        return np.nan
def calculate_area_under_graph(data_list, dt):
    #we try to find the area under graph
    data_list = safe_unwrap(data_list)  # making the check about the data list
    if data_list is not None:
        abs_vector = np.abs(data_list)
        area = np.sum(abs_vector*dt)
        return area
    else:
        return np.nan
def calculate_peak_to_peak_time_variables (data_list, dt, absolute = True):
    from scipy.signal import find_peaks
    data_list = safe_unwrap(data_list)  # making the check about the data list
    if data_list is not None:
        if absolute:
            #this will be used for finding also the time between peaks including lows, which not showing the frequencies but rather the change rate
            data_list = np.abs(data_list)
        peaks_indices, _ = find_peaks(data_list, distance=5)
        if len(peaks_indices) <2:
            #if there is only one peak - we return the entire window time
            return len(data_list) * dt
        time_differences = np.diff(peaks_indices) * dt #we find the indices differences between them and then normalize it to time
        return np.mean(time_differences)
    else:
        return np.nan

def add_time_dependent_features(df, column_list, num_features):
    #this function meant for finding the more complex time dependent features mentioned abouve -Slew Rate, Area Under Graph
    new_columns = {}
    for column in column_list:
        # For each sensor, we will calculate each metric for every axis, and also for the magnitude of all the axes combined
        dt = 0.2
        if 'Mag' in column:
            #as the sampling rate is different
            dt = 0.4
        #adding area under graph
        new_columns[column +"_area_under_graph"] = df[column].apply(lambda x: calculate_area_under_graph(x, dt)).values
        # print(new_columns[column + '_area_under_graph'])
        num_features += 1

        #adding slew rate
        new_columns[column +"_slew_rate"] = df[column].apply(calculate_slew_rate).values
        # print(new_columns[column + '_slew_rate'])
        num_features += 1

        #adding mean time between peaks
        new_columns[column + "mean_peak_to_peak_time"] = df[column].apply(lambda x: calculate_peak_to_peak_time_variables(x, dt, absolute=False)).values

        #adding mean time between peaks and lows - helpful for non harmonic behaviour
        new_columns[column + "mean_abs_peak_to_abs_peak_time"] = df[column].apply(lambda x: calculate_peak_to_peak_time_variables(x, dt, absolute=True)).values

    # Now we concatenate the newly created dict to the df - just one addition
    df_new = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df_new, num_features


##-------statistical-based features-------##
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

def compute_AbsCV(data_list):
    #Here, we calculate coefficient of variation of the absolute value of the signal.
    #helps the model identify signals with similar power but different variety
    data_list = safe_unwrap(data_list)
    if data_list is not None:
        data_list = np.abs(data_list)
        mean_abs = np.mean(data_list)
        std_abs = np.std(data_list)
        if mean_abs == 0:
            return 0.0
        else:
            return mean_abs/std_abs
    else:
        return np.nan

def add_disribution_features(df, column_list, num_features):
    #This function is meant to find features which are connected to the distribution.
    #it receives as input a dataframe to conduct the calculation on and to add the features to,
    #a list of columns to calculate based on them, and at last - num_features that will be used for tracking the number of features addded.

    # in order to achieve better running time, we will add a dict containing all the new columns, and then add them together
    new_columns = {}
    for column in column_list:
        new_columns[column + '_skewness'] = df[column].apply(compute_skewness)
        print(f"added {column + '_skewness'} column")
        # print(new_columns[column + '_skewness'])
        num_features += 1
        new_columns[column + '_kurtosis'] = df[column].apply(compute_kurtosis)
        print(f"added {column + '_kurtosis'} column")
        # print(new_columns[column + '_kurtosis'])
        num_features += 1
        new_columns[column + "_AbsCV"] = df[column].apply(compute_AbsCV)
        print(f"added {column + '_AbsCV'} column")

        num_features += 1
    #Now we concatenate the newly created dict to the df - just one addition
    df_new = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df_new, num_features

def calculate_cusum(series, target, slack):
    #CUSUM is a metric intended to find how the mean changes over time.
    #CUSUM accumulates the small deviations from the expected average, and when this accumulation reaches a threshold, it signals that the process average is shifting.
    #COSUM hint how the data fluctuate over time and by that reflects the change in movement
    #Here, the function recieves PD series, target which represents a traget mean, and a slack which is the reference value.
    # target
    cusum_list = []
    current_cusum = 0
    for value in series:
        #Here we find the COSUM in order to track rise in it.
        #to track fall in COSUM, we will add -value + target - slack
        current_cusum = max(0, current_cusum + (value - target) - slack)
        cusum_list.append(current_cusum)
    return cusum_list

def add_Cosum_metrics (df, column):
    #this function calculate the Cosum of a df[column].

    df = df.copy()  # In order to prevent warnings

    #we create the columns that will be added
    df[column+'_Mean_Shift'] = np.nan
    df[column+'_CUSUM+_Feature'] = np.nan
    df[column+'_CUSUM-_Feature'] = np.nan

    #Cosum is tracking the movement in the window, but we do not want to compare separate recordings.
    # This metric is preformed on a normalize data and each recording was normalize separately, so we group each recording by using its identifiers.
    grouped = df.groupby(['Group number', 'Participant ID', 'Recording number'])
    for (group, pid, rec), new_df in grouped:
        #We get the median of the mean column
        median_val = np.median(new_df[column + "_mean"])
        # As the handwashing event is relatively rare, the STD may not be ideal.
        # Thus we use MAD- Median Absolute Deviation. the slack will be 0.5 of that
        median_abs_deviation = np.median(np.abs(new_df[column + "_mean"] - median_val))

        # In standard deviation, 1.486 is the ratio between Std to MAD
        ROBUST_STD = median_abs_deviation * 1.4826

        # we define the STD
        K_slack = 0.5 * ROBUST_STD

        # here we calculate the mean shift and the stdshift between adjacent time points
        mean_shift = new_df[column + "_mean"] - new_df[column + "_mean"].shift(1).fillna(new_df[column + "_mean"].iloc[0])
        std_relative_shift = (new_df[column + "_std"] - new_df[column + "_std"].shift(1)) / new_df[column + "_std"].shift(1).fillna(new_df[column + "_std"].iloc[0])

        # We find the Cosum based on the "mean" column, with the found slack and target = 0 -
        # the data is normalized so the overall average mean should be 0 - this is the target.
        cusum_pos_values = calculate_cusum(
            new_df[column + "_mean"],
            0,
            K_slack
        )
        if not "SM" in column:
            # we will not get under the zero values which makes it irrelevant
            cusum_neg_values = calculate_cusum(
                -new_df[column + "_mean"],
                -0,
                K_slack
            )

        # We insert the data to the df
        df.loc[new_df.index, column+'_CUSUM+_Feature'] = cusum_pos_values
        if not "SM" in column:
            df.loc[new_df.index, column+'_CUSUM-_Feature'] = cusum_neg_values
        df.loc[new_df.index, column+'_Mean_Shift'] = mean_shift
        df.loc[new_df.index, column+'_Relative_STD_Shift'] = std_relative_shift

    print(f"added {column} COSUM metrics")


    return df


##-------Frequency-domain features-------##

def PSD_for_signal(signal, sampling_rate=50):
    from scipy.signal import welch
    # This function receives a signal and calculate its PSD and frequencies.
    # PSD is the measure of signal's power content versus frequency, and it is more useful for cases where the signal is noisy and not harmonic.
    frequencies, psd = welch(signal, fs=sampling_rate, nperseg=len(signal))
    # nperseg decides how many samples will be calculated in each iteration of PSD analyzing.
    # we take the number of samples in the window.
    # we get as output the vector of frequencies and the vector of PSD values that fit the matching frequency
    return frequencies, psd


def calculate_frequency_domain_features(data_list, sampling_rate=50):
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

        #Let's calculate kurtosis and skewness of the PSD also to examine its distribution
        psd_skewness = compute_skewness(norm_psd)
        psd_kurtosis = compute_kurtosis(norm_psd)

    return spectral_entropy, total_energy, freq_centroid, dominant_freq, freq_variance, psd_skewness, psd_kurtosis

def add_frequency_domain_features(df, column_list, num_features):
    # This function is meant to find features dependent on the frequency domain.
    # it receives as input a dataframe to conduct the calculation on and to add the features to,
    # a list of columns to calculate based on them, and at last - num_features that will be used for tracking the number of features addded.

    new_columns = {}
    feature_suffixes = ['spectral_entropy', 'total_energy', 'frequency_centroid',
                        'dominant_frequency', 'frequency_variance','psd_skewness' , 'psd_kurtosis']
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

#------ Derivative features------##

def compute_derivatives(data_list, sampling_rate=50):
    # We add metrics which are connected to the derivative, whose physical interpretation is the velocity and acceleration.
    # By looking at the velocity and acceleration, we can learn about the change in movement during the period we are looking at.
    data_list = safe_unwrap(data_list)
    if data_list is not None:
        # we receive the data of this window and the sampling rate, and calculate the first and second derivative - the velocity and acceleration.
        velocity =  np.diff(data_list) / (1/sampling_rate)
        acceleration = np.diff(velocity) / (1/sampling_rate)

        #we find the standard deviation of the movement to see how the velocity and acceleration changed during the window
        velocity_std = np.std(velocity)
        acceleration_std = np.std(acceleration)

        # We find the median of the velocity and acceleration, as it is unbiased metric to evaluate the middle point.
        velocity_median = np.median(velocity)
        acceleration_median = np.median(acceleration)

        # We find the kurtosis, as it allows to understand whether there were more changes in the negative or the positive direction in the velocity and the jerk
        velocity_kurtosis = compute_kurtosis(velocity)
        acceleration_kurtosis = compute_kurtosis(acceleration)


        return velocity_std, acceleration_std,  velocity_median, acceleration_median, velocity_kurtosis, acceleration_kurtosis


    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def add_derivative_features(df, column_list, num_features):
    #Here, we just add the derivatives' features for every axis and the combined magnitude for the Acc and Gyro.
    # The magnometer is not relevant for these features.
    new_columns = {}

    feature_suffixes = ['velocity_std', 'jerk_std', 'velocity_median',
                        'jerk_median', 'velocity_kurtosis','jerk_kurtosis']
    for column in column_list:
        if "MAG" not in column:
            features_series = df[column].apply(
                lambda x: pd.Series(
                    compute_derivatives(x, sampling_rate= 50),
                    index=feature_suffixes  # defining the name of returned columns
                )
            )
        # adding to the dict
        for suffix in feature_suffixes:
            col_name = f"{column}_{suffix}"
            new_columns[col_name] = features_series[suffix]
            print(f"added {col_name} column")

        num_features += len(feature_suffixes)
    df_new = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df_new, num_features

##-------EMD features - wasn't used at the end-------##

def find_imfs_properties(data_list):
    #It seems like the main factors are the std and relative energy of the imfs, so this will be our focus
    data_list = safe_unwrap(data_list)
    emd = PyEMD.EMD()
    #we find the list of Imfs
    imfs = emd(data_list)
    # we check what happen where there are no IMFs
    if imfs.shape[0] == 0:
        return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
    #We compute the total enery of all the IMFs
    total_energy = np.sum(imfs ** 2)

    if len(imfs) > 1:
        imfs_std = []
        relative_imf_energy = []
        for i in range(2):
            #we find the standard deviation of each IMF
            imfs_std.append(np.std(imfs[i,:]))
            # we find the relative energy of each IMF - it can show how this imf is significant in relation to other
            imf_energy = np.sum(imfs[i,:] ** 2)
            relative_imf_energy.append(imf_energy / total_energy)
        return pd.Series([imfs_std[0],imfs_std[1], relative_imf_energy[0], relative_imf_energy[1], total_energy])
    else:
        #Now we deal with the case of only one imf - which may also indicate about the activity and be helpful
        imf_std = np.std(imfs[0,:])
        imf_energy = np.sum(imfs[0,:] ** 2)
        relative_imf_energy = (imf_energy / total_energy)
        #we choose to pass zero and not NaN in order to be able to let the model distinguish based on numbers,
        # as we assume no IMF will get energy and std == 0 (and if so - it won't be the handwashing)
        return pd.Series([imf_std, 0, relative_imf_energy, 0, total_energy])



def EMD_properties(df, column_list, num_features, n_jobs=-1) :
    #Now - we try to find EMD properties that will help us.
    #We use only the first and second IMFs, as we don't want to take all the Imfs to reduce complexity
    #and on the hand - those are the functions that express the changes the most.
    new_columns = {}
    feature_suffixes = ['imf1 std', 'imf2 std', 'imf1 relative energy',
                        'imf2 relative energy', 'total_EMD_energy']


    for column in column_list:
        # For each sensor, we will calculate each metric for every axis, and also for the magnitude of all the axes combined
        #As we have seen those features are extremely time consuming, we use Parallel to operate parallel searches and by that accelerate the calculations
        # results_list = Parallel(n_jobs=n_jobs)(
        #     delayed(find_imfs_properties)(data) for data in df[column]
        # )
        results_list = [find_imfs_properties(data) for data in df[column]]

        features_df = pd.DataFrame(results_list, index=df[column].index, columns=feature_suffixes)

        # adding to the dict
        for suffix in feature_suffixes:
            col_name = f"{column}_{suffix}"
            new_columns[col_name] = features_df[suffix]
            print(f"added {col_name} column")

        num_features += len(feature_suffixes)
    df_new = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df_new, num_features

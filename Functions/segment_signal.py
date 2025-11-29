import pandas as pd
import numpy as np

from .load_data import load_data

# Window step is the delay between each window. it can be chosen in various ways, for overlapping or not.
def segment_signal(data_path, window_size, window_step):
    print('============================ Segmenting signal again :)))')
    data_files = load_data(data_path)

    hand_washing_duration_list = []
    keys = data_files.keys()
    for key in data_files.keys():
        recording = data_files[key]['label']['data']
        # we get the label table
        if recording['End (Seconds from Recording Start)'.upper()].iloc[-1] < 330:
            data_files[key]["Protocol"] += 1
        # now let's have only the phases of hand washing. a check is conducted to make sure no miss-labeling
        relevant_phases = recording[(recording["Label".upper()] == 1) & (recording['Description'.upper()].isin(
            ['HandWashing', 'Hand Washing', 'Handwashing', 'washing', 'Washing']))]
        phases_by_name = recording[(recording['Description'.upper()].isin(
            ['HandWashing', 'Hand Washing', 'Handwashing', 'washing', 'Washing']))]
        phases_by_number = recording[(recording["Label".upper()] == 1)]
        if not ((relevant_phases.equals(phases_by_name)) or (relevant_phases.equals(phases_by_number))):
            raise ValueError(
                f"invalid recording {data_files[key]['Group number']}_{data_files[key]['Recording number']}_{data_files[key]['Participant ID']} - Label do not match actual action")

        for index, row in relevant_phases.iterrows():
            # we extract the times of handwashing activities, and also see the Mean and Std to effectively declare windows
            start_col = 'Start (Seconds from Recording Start)'.upper()
            end_col = 'End (Seconds from Recording Start)'.upper()
            start = float(row[start_col])
            end = float(row[end_col])

            total_duration = end - start

            hand_washing_duration_list.append(total_duration)
            data_files[key]['Handwashing time'].append((start, end))

    # print(len(hand_washing_duration_list))
    print(f"the average hand washing duration is: {np.mean(hand_washing_duration_list):3.2f}")
    print(f"the standard deviation of hand washing duration is: {np.std(hand_washing_duration_list):3.2f}")

    def create_window(window_size, window_step, recording):
        # at first, we commit check that each recording starts with 0. if not, we "normalize" the data by creating a reset to zero
        for sensor in ["Acc", "Gyro", "Mag"]:
            starting_point = recording[sensor]['data']['elapsed (s)'.upper()].min()
            if starting_point != 0:
                for index in time_column.index.tolist():
                    recording[sensor]['data']['elapsed (s)'.upper()][index] -= starting_point
                    for time_tuple in recording['Handwashing time']:
                        # if the data table also do not start from zero.
                        time_tuple[0] -= starting_point
                        time_tuple[1] -= starting_point
            time_column = recording[sensor]['data']['elapsed (s)'.upper()]
            recording_start = time_column.min()
            recording_end = time_column.max()

            window_starting_time = 0

        window_starting_points = []
        window_ending_points = []

        while window_starting_time + window_step <= recording_end:
            window_starting_points.append(window_starting_time)
            window_ending_time = window_starting_time + window_size
            if window_ending_time > recording_end:
                # in case we pass the limit of the end of recording with the new end of window,
                # the end point will be the actual ending of the recording
                window_ending_points.append(recording_end)
                break
            window_ending_points.append(window_ending_time)
            window_starting_time += window_step

        return window_starting_points, window_ending_points

    def assign_label_to_window(window_starting_time, window_ending_time, reference_timing_list):
        # the intention of this function is to assign the window into a certain label.
        # as rule of decision, we decided that if the majority of the window (meaning 50% or above) is Handwashing, it will be considered as handwashing. other wise, it will be labeled as zero.
        window_length = window_ending_time - window_starting_time
        # we try to find the overlap, which there can be 4 cases:
        # a. window is completely included in the period between the time points
        # b. the period between the time points includes the entire window
        # c. there is an overlap that starts with the start of time point periods and ends with the window's end
        # d. there is an overlap that starts with the start of the window periods and ends with the end point of the period
        # therefore, we calculate overlap in a way that take all of them into consideration, and assign the label if the overlap is at leat 50% of the window
        for timing in reference_timing_list:
            overlap_start = max(window_starting_time, timing[0])
            overlap_end = min(window_ending_time, timing[1])
            overlap_length = max(0, overlap_end - overlap_start)
            if overlap_length >= 0.5 * window_length:
                return 1
        return 0

    columns_names = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number',
                     'Recording number', 'Protocol', 'Label']
    X_matrix = pd.DataFrame([], columns=columns_names)

    # We now go over each recording and creates the windows, and labeling each one of them
    for recording in data_files.values():
        window_starting_points, window_ending_points = create_window(window_size, window_step, recording)
        for i in range(len(window_starting_points)):  # We go over each window
            window_starting_time = window_starting_points[i]
            window_ending_time = window_ending_points[i]
            label = assign_label_to_window(window_starting_time=window_starting_time,
                                           window_ending_time=window_ending_time,
                                           referance_timing_list=recording['Handwashing time'])

            # Now, creating the dict that will be added to the DF
            row_dict = {'First second of the activity': window_starting_time,
                        'Last second of the activity': window_ending_time,
                        'Participant ID': recording['Participant ID'], 'Group number': recording['Group number'],
                        'Recording number': recording['Recording number'], 'Protocol': recording['Protocol'],
                        'Label': label}
            X_matrix.loc[len(X_matrix.index)] = row_dict
            print(f"row {len(X_matrix.index) - 1} has been added")

    Y_vector = X_matrix['Label']
    X_matrix = X_matrix.drop(columns=['Label'])

    # print(X_matrix[(X_matrix["First second of the activity"] != 0) &(X_matrix["Last second of the activity"] != 0)][:1000])
    # print(Y_vector)

    return X_matrix, Y_vector


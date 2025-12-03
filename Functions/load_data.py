import pandas as pd
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

#
# ##As loading data happens in both Part A and Part B, we define a function named Load data.
# #This function get data path and return dict that include all of the CSV data of recording from all sensor, alongside the identifiers
# #We Load the data just once - avoid loading each time which damage the run time.
# def load_data(data_path):
#     all_items = os.listdir(data_path)
#     data_files = {}
#     # we run over the data file
#     for item in all_items:
#         folder_path = os.path.join(data_path, item)
#         #we run over each subfile representing subgroups
#         for file_path in os.listdir(folder_path):
#     #for file_path in all_items:
#             #This implementation is done to make sure no data will be missed due to suffix of CSV instead of csv
#             if file_path.lower().endswith('.csv') and os.path.isfile(os.path.join(folder_path, file_path)):
#                 file_name = os.path.basename(file_path)
#
#                 # Here, we create for each record a dict that includes the Recording number, the group number, Participant ID
#                 # and another identifier for the record type which is a dict of the actual data and the record type.
#                 #this will help differntiate between data_types
#                 recording_identifier = file_name[:7]
#                 list_of_elements = file_name.split('_')
#                 recording_type = list_of_elements[3].removesuffix(".csv")
#                 psv_data = pd.read_csv(os.path.join(folder_path, file_path))
#                 psv_data.columns = psv_data.columns.str.upper()
#                 psv_data = psv_data.dropna()
#
#                 data_dict = {"Recording type": recording_type, "data": psv_data}
#
#                 if data_files.get(recording_identifier):
#                     data_files[recording_identifier][recording_type] = data_dict
#                 else:
#                     data_files[recording_identifier] = {}
#                     data_files[recording_identifier]["Group number"] = list_of_elements[0]
#                     data_files[recording_identifier]["Recording number"] = list_of_elements[1]
#                     data_files[recording_identifier]["Participant ID"] = list_of_elements[2]
#                     data_files[recording_identifier][recording_type] = data_dict
#                     # A list that will get the time of handwashing. It is a list in case there are more than one handwashing events in the recording
#                     data_files[recording_identifier]['Handwashing time'] = []
#                     #the next one will help us to distinguish between protocol recordings to regular one
#                     data_files[recording_identifier]['Protocol'] = 0
#     return data_files


##As loading data happens in both Part A and Part B, we define a function named Load data.
#This function get data path and return dict that include all of the CSV data of recording from all sensor, alongside the identifiers
#We Load the data just once - avoid loading each time which damage the run time.

#This function is meant for the reading of the CSV files, so we can read them in parallel and by that reduce time
def read_csv_parallel(path):
    df = pd.read_csv(path)
    mapping_dict = {
        'START': 'START (SECONDS FROM RECORDING START)',
        'END': 'END (SECONDS FROM RECORDING START)',
        'START(S)': 'START (SECONDS FROM RECORDING START)',
        'END(S)': 'END (SECONDS FROM RECORDING START)',
        'START (S)': 'START (SECONDS FROM RECORDING START)',
        'END (S)': 'END (SECONDS FROM RECORDING START)',
        'DESCRIBTION': 'DESCRIPTION',
        'START (SECONDS FROM RECORDING START) ': 'START (SECONDS FROM RECORDING START)',
    'END (SECONDS FROM RECORDING START) ': 'END (SECONDS FROM RECORDING START)'
    }

    #All of those - meant for dealing with inconsistency in naming in comparison to the instruction (cases we encountered)
    df.columns = df.columns.str.replace('\n', ' ').str.upper()
    df.columns = [mapping_dict.get(col, col) for col in df.columns]

    df = df.dropna()
    return path, df

def load_data(data_path):
    all_items = os.listdir(data_path)
    data_files = {}

    # Collect all full CSV paths
    csv_paths = []
    # we run over the data file
    for item in all_items:
        folder_path = os.path.join(data_path, item)
        # As the data is structured in folder, we make sure the item we are viewing is a folder. if not - we move to the next one
        if not os.path.isdir(folder_path):
            continue

        #we run over each subfile representing subgroups
        for file_path in os.listdir(folder_path):
            full_file_path = os.path.join(folder_path, file_path)
            #This implementation is done to make sure no data will be missed due to suffix of CSV instead of csv
            if file_path.lower().endswith(".csv") and os.path.isfile(full_file_path):
                # we append it to the CSV path list
                csv_paths.append(full_file_path)

    # Load all CSV files in parallel
    loaded = {}
    #Here, we run a parallel process to load all the data psv files for later use
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(read_csv_parallel, p): p for p in csv_paths}
        for future in as_completed(futures):
            full_path, df = future.result()
            loaded[full_path] = df

    # now we use the already loaded dfs
    for item in all_items:
        folder_path = os.path.join(data_path, item)
        if not os.path.isdir(folder_path):
            continue
        for file_path in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_path)

            if file_path.lower().endswith('.csv') and os.path.isfile(full_path):
                file_name = os.path.basename(file_path)
                print(file_name)

                # Here, we extract the basic identifires of the data.


                #it will be the group_number_recording_number_participant_ID
                recording_identifier = file_name[:7]
                # we split the file name into 3 parts
                list_of_elements = file_name.split('_')
                #here we will get the sensor name, and dealing with observed mismatches
                recording_type = list_of_elements[3].removesuffix(".csv")
                if recording_type.upper() == 'Label'.upper():
                    recording_type = recording_type.lower()
                elif recording_type.upper() == 'ACC':
                    recording_type = 'Acc'
                elif recording_type.upper() == 'GYRO':
                    recording_type = 'Gyro'
                elif recording_type.upper() == 'MAG':
                    recording_type = 'Mag'
                # we use the preloaded DataFrame
                psv_data = loaded[full_path]

                # These identifies will be used for a creation of a dict for each record, that includes the Recording number, the group number, Participant ID
                # and another identifier for the record type which is a dict of the actual data and the record type.
                # this will help differentiate between data_types

                data_dict = {"Recording type": recording_type, "data": psv_data}

                #it there is already a suitable dict
                if data_files.get(recording_identifier):
                    data_files[recording_identifier][recording_type] = data_dict
                # it it is the first dat of this type that is added
                else:
                    #we create the actual dict
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
#
# data_path = r'C:\Users\nirei\PycharmProjects\Continous monitoring\data'
#
# if __name__ == "__main__":
#     start = time.time()
#     d_original = load_data(data_path)
#     end_time = time.time()
#     print (f"time for original:{end_time - start} seconds")
#     start = time.time()
#     d_parallel = load_data_parallel(data_path)
#     end_time = time.time()
#     print (f"time for parallel:{end_time - start} seconds")
#
#     def compare_data_structures(d1, d2):
#         # השוואת כל ה-recording identifiers
#         if set(d1.keys()) != set(d2.keys()):
#             print("❌ Different recording identifiers!")
#             print("Only in d1:", set(d1.keys()) - set(d2.keys()))
#             print("Only in d2:", set(d2.keys()) - set(d1.keys()))
#             return False
#
#         # השוואה בתוך כל recording_identifier
#         for rec_id in d1.keys():
#             keys1 = set(d1[rec_id].keys())
#             keys2 = set(d2[rec_id].keys())
#
#             if keys1 != keys2:
#                 print(f"❌ Different keys inside {rec_id}")
#                 print("Only in original:", keys1 - keys2)
#                 print("Only in parallel:", keys2 - keys1)
#                 return False
#
#             # השוואת כל ה-dataframes
#             for key in keys1:
#                 if key in ["Group number", "Recording number", "Participant ID", "Handwashing time", "Protocol"]:
#                     # שדות פשוטים — רק משווים ערכים
#                     if d1[rec_id][key] != d2[rec_id][key]:
#                         print(f"❌ Value mismatch in metadata field '{key}' of {rec_id}")
#                         return False
#                 else:
#                     # זה רישום של חיישן — צריך להשוות DataFrame
#                     df1 = d1[rec_id][key]["data"]
#                     df2 = d2[rec_id][key]["data"]
#
#                     if not df1.equals(df2):
#                         print(f"❌ DataFrame mismatch in {rec_id} / {key}")
#                         print("Shape original:", df1.shape, "Shape parallel:", df2.shape)
#                         return False
#
#         print("✅ All matched! Output is identical.")
#         return True
#
#
#     compare_data_structures(d_original, d_parallel)
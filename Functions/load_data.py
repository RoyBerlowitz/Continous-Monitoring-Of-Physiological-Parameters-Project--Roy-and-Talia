import pandas as pd
import os

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

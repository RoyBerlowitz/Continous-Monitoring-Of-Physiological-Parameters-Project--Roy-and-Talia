import pandas as pd
import os

from Functions import segment_signal, extract_features

#cosnts
segment_signal_x_filename = "X_matrix_1.pkl"
segment_signal_y_filename = "Y_matrix_1.pkl"

def run_part_a(data_path, use_csv=False, remake_csv=False):
    ##-------Part A: Segmentation-------
    X_matrix_1, Y_matrix_1 = None, None

    if use_csv and not remake_csv:
        if os.path.exists(segment_signal_x_filename):
            X_matrix_1 = pd.read_pickle(segment_signal_x_filename)
        if os.path.exists(segment_signal_y_filename):
            Y_matrix_1 = pd.read_pickle(segment_signal_y_filename)

    if X_matrix_1 is None or Y_matrix_1 is None:
        X_matrix_1, Y_vector_1 = segment_signal(data_path, 3, 1.5)

        if use_csv:
            X_matrix_1.to_pickle(segment_signal_x_filename)
            Y_vector_1.to_pickle(segment_signal_y_filename)

    # return
    ##-------Part B: Feaature Extraction-------
    X_features = extract_features(data_path, X_matrix_1)
    X_features.to_excel("X_features.xlsx", index=False)

data_path = r"C:\Users\nirei\OneDrive\Desktop\Bachelors Degree - Biomedical Engineering And Neuroscience\Year 4\Semester A\Continuous Monitoring of Physiological Parameters\PythonProject7\02"
# data_path = r"/Users/talia/Downloads/02 copy 3"
# run_part_a(data_path, use_csv=True, remake_csv=False)
run_part_a(data_path)


##-------Part A: Segmentation-------

# Window step is the delay between each window. Because the

#We call the function 3 times to get 3 window sizes
# X_matrix_1, Y_vector_1 = segment_signal(data_path, 3, 1.5)
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

##-------Part C: Train & Test -------

##-------Part D: Feature Correlation Analysis -------



import pandas as pd
import os

from Functions import segment_signal, extract_features, split_data, load_cache_or_compute

#cosnts
#todo change before handing in
is_dev = True #False

def run_part_a(data_path, force_recompute_seg=True, force_recompute_features=True, force_recompute_splits=True):
    ##--------------- Part A: Segmentation ----------------##
    X_matrix, Y_vector = load_cache_or_compute(
        "segment_output.pkl",
        lambda: segment_signal(data_path, 3, 1.5),
        force_recompute=force_recompute_seg,
        save=is_dev
    )
    ##--------------- Part B: Feature Extraction -----------##
    X_features = load_cache_or_compute(
        "X_features.pkl",
        lambda: extract_features(data_path, X_matrix),
        force_recompute=force_recompute_features,
        save=is_dev
    )

    ##-------Part C: Train & Test -------
    splits = load_cache_or_compute(
        "splits.pkl",
        lambda: split_data(X_features, Y_vector),
        force_recompute=force_recompute_splits,
        save=is_dev
    )

data_path = r"C:\Users\nirei\OneDrive\Desktop\Bachelors Degree - Biomedical Engineering And Neuroscience\Year 4\Semester A\Continuous Monitoring of Physiological Parameters\PythonProject7\02"
data_path = r"/Users/talia/Downloads/02 copy 3"
run_part_a(data_path, force_recompute_seg=False, force_recompute_features=False, force_recompute_splits=True)
# run_part_a(data_path)


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



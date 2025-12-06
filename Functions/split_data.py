from sklearn.model_selection import train_test_split
import pandas as pd

from .split_data_helper_functions import *

def split_data(X_features, Y_samples):
    """
       Split the dataset using two strategies: Within-Group and Leave-Group-Out.

       Parameters
       ----------
       X_features : pandas.DataFrame
           Feature matrix containing participant metadata such as Participant ID,
           Group number, and Protocol/Routine labels.
       Y_samples : pandas.Series or pandas.DataFrame
           Corresponding labels for each row in `X_features`.

       Returns
       -------
       list
           A two-element list containing:

           1. **Split 1 — Within-Group Results**
              A list of lists, where each inner list corresponds to one participant
              and has the form:
                  [X_train, X_test, Y_train, Y_test]

              - Train = all Protocol recordings + 80% of that participant's Routine data
              - Test  = remaining 20% of that participant's Routine data

           2. **Split 2 — Leave-Group-Out Results**
              A single list with the form:
                  [X_train, X_test, Y_train, Y_test]

              - Train = all data (Protocol + Routine) from 80% of the participants
              - Test  = Routine-only data from the remaining 20% of participants
              - This split evaluates generalization to entirely unseen participants.
       """
    #In general a participant is participant_ID+group_numer
    #consts
    participant_id_column_name = 'Participant ID'
    group_number_column_name = 'Group number'
    protocol_column_name = 'Protocol'
    recording_number_column_name = 'Recording number'

#split 1 Within-Group Split
    # • Train: For each group, use ALL "Protocol" recordings + 80% of the "Routine"
    # recordings.
    # • Test: Use the remaining 20% of the "Routine" recordings
    #for each participant split, train= all protocol +80% of left over, test= 20% of leftover
    all_participants = list(X_features.groupby([participant_id_column_name, group_number_column_name]).groups.keys())
    split1_X_trains = []
    split1_X_tests = []
    split1_Y_trains = []
    split1_Y_tests = []

    for (pid, gid) in all_participants:
        X_train_list = []
        X_test_list = []
        Y_train_list = []
        Y_test_list = []

        # Filter only this participant+group rows
        mask = ((X_features[participant_id_column_name] == pid) &
                (X_features[group_number_column_name] == gid))

        df_p = X_features[mask]
        y_p = Y_samples[mask]

        # --- Add all Protocol recordings to train ---
        df_protocol = df_p[df_p[protocol_column_name] == 1]
        y_protocol = y_p[df_p[protocol_column_name] == 1]

        X_train_list.append(df_protocol)
        Y_train_list.append(y_protocol)

        # --- Split Routine (Protocol==0) into train/test ---
        df_routine = df_p[df_p[protocol_column_name] == 0]
        y_routine = y_p[df_p[protocol_column_name] == 0]

        if len(df_routine) > 0:

            X_train_r, X_test_r, Y_train_r, Y_test_r = split_by_recording(
                df_routine,
                y_routine,
                recording_col=recording_number_column_name,
                test_size=0.2
            )

            X_train_list.append(X_train_r)
            Y_train_list.append(Y_train_r)

            X_test_list.append(X_test_r)
            Y_test_list.append(Y_test_r)

        # --- Combine all sub-results and add to mega list ---
        split1_X_trains.append(pd.concat(X_train_list, ignore_index=True))
        split1_X_tests.append(pd.concat(X_test_list, ignore_index=True))
        split1_Y_trains.append(pd.concat(Y_train_list, ignore_index=True))
        split1_Y_tests.append(pd.concat(Y_test_list, ignore_index=True))

    split1 = [split1_X_trains,split1_X_tests,split1_Y_trains,split1_Y_tests]



#split 2 Leave-Group-Out
    # 80% of participants go to TRAIN, 20% to TEST
    # For train participants → include all their rows (Protocol + Routine)
    # For test participants → include only Routine rows (Protocol = 0)

    # Create a combined participant identifier
    df = X_features #.copy()
    df["__participant_key__"] = df[participant_id_column_name].astype(str) + "_" + df[group_number_column_name].astype(str)
    participant_keys = df["__participant_key__"].unique()

    train_participants, test_participants = train_test_split(
        participant_keys,
        test_size=0.2,
        random_state=42
    )

    # TRAIN: all rows belonging to train participants
    train_mask = df["__participant_key__"].isin(train_participants)
    X_train = df[train_mask]
    y_train = Y_samples[train_mask]

    # TEST: only routine rows of test participants
    test_mask = df["__participant_key__"].isin(test_participants) & (df[protocol_column_name] == 0)
    X_test = df[test_mask]
    y_test = Y_samples[test_mask]

    # Clean up before returning
    X_train = X_train.drop(columns="__participant_key__")
    X_test = X_test.drop(columns="__participant_key__")

    split2 = [X_train,X_test,y_train,y_test]

    return [split1,split2]
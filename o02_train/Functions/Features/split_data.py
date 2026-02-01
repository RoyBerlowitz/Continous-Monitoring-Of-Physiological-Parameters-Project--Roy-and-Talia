from sklearn.model_selection import train_test_split

def split_data(X_features, Y_samples):
    # consts
    participant_id_column_name = 'Participant ID'
    group_number_column_name = 'Group number'
    protocol_column_name = 'Protocol'

    # split 2 Leave-Group-Out
    # 80% of participants(!!!! not groups) go to TRAIN, 20% to TEST
    # For train participants → include all their rows (Protocol + Routine)
    # For test participants → include only Routine rows (Protocol = 0)

    # Create a combined participant identifier
    df = X_features
    df["__participant_key__"] = df[participant_id_column_name].astype(str) + "_" + df[group_number_column_name].astype(
        str)
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

    return [X_train, X_test, y_train, y_test]

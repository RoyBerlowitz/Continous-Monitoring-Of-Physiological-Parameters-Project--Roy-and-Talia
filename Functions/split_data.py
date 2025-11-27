from sklearn.model_selection import train_test_split
import numpy as np

def split_data(X_features, Y_samples): #, groups, recording_type):

    """
    Split the dataset in two ways:
        1. Within-Group Split
        2. Leave-Group-Out Split

    Parameters
    ----------
    X_features : np.ndarray
        Feature matrix of shape (N_samples, N_features)
    Y_samples : np.ndarray
        Labels vector of shape (N_samples,)
    groups : np.ndarray
        Group (user) ID per sample, shape (N_samples,)
    recording_type : np.ndarray
        Array of strings, "Protocol" or "Routine" per sample.

    Returns
    -------
    split1 : dict
        {
            'X_train', 'X_test', 'Y_train', 'Y_test'
        }

    split2 : dict
        {
            'X_train', 'X_test', 'Y_train', 'Y_test'
        }
    """

    # =========================================================================
    # ---------------------- SPLIT 1: WITHIN-GROUP SPLIT ----------------------
    # =========================================================================

    # Containers for final training/testing sets
    X_train_1, X_test_1, Y_train_1, Y_test_1 = [], [], [], []
    return
    # Iterate over each unique group (participant)
    for g in np.unique(groups):

        # ---- Extract samples belonging to this group ----
        idx_group = (groups == g)

        X_g = X_features[idx_group]
        Y_g = Y_samples[idx_group]
        types_g = recording_type[idx_group]

        # ---- Separate Protocol vs Routine ----
        X_protocol = X_g[types_g == "Protocol"]
        Y_protocol = Y_g[types_g == "Protocol"]

        X_routine = X_g[types_g == "Routine"]
        Y_routine = Y_g[types_g == "Routine"]

        # ---- Stratified 80/20 split for Routine only ----
        if len(Y_routine) > 0:
            X_r_train, X_r_test, Y_r_train, Y_r_test = train_test_split(
                X_routine,
                Y_routine,
                test_size=0.2,
                stratify=Y_routine,
                random_state=42
            )
        else:
            X_r_train, X_r_test, Y_r_train, Y_r_test = [], [], [], []

        # ---- Build final group-specific splits ----
        # Training: ALL Protocol + 80% Routine
        X_train_1.append(np.vstack([X_protocol, X_r_train]) if len(X_protocol) else X_r_train)
        Y_train_1.append(np.hstack([Y_protocol, Y_r_train]) if len(Y_protocol) else Y_r_train)

        # Testing: remaining 20% Routine
        if len(X_r_test):
            X_test_1.append(X_r_test)
            Y_test_1.append(Y_r_test)

    # ---- Concatenate all groups ----
    X_train_1 = np.vstack(X_train_1)
    Y_train_1 = np.hstack(Y_train_1)
    X_test_1 = np.vstack(X_test_1)
    Y_test_1 = np.hstack(Y_test_1)

    split1 = {
        "X_train": X_train_1,
        "X_test": X_test_1,
        "Y_train": Y_train_1,
        "Y_test": Y_test_1
    }

    # =========================================================================
    # --------------------- SPLIT 2: LEAVE-GROUP-OUT SPLIT --------------------
    # =========================================================================

    unique_groups = np.unique(groups)

    # ---- Stratified split but on groups ----
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=0.2,
        random_state=42
    )

    # ---- Build training set (ALL data for selected groups) ----
    idx_train_groups = np.isin(groups, train_groups)
    idx_test_groups  = np.isin(groups, test_groups)

    # Training data includes BOTH Protocol + Routine
    X_train_2 = X_features[idx_train_groups]
    Y_train_2 = Y_samples[idx_train_groups]

    # Testing data: ONLY routine samples from held-out groups
    idx_routine = (recording_type == "Routine")
    idx_test_final = idx_test_groups & idx_routine

    X_test_2 = X_features[idx_test_final]
    Y_test_2 = Y_samples[idx_test_final]

    split2 = {
        "X_train": X_train_2,
        "X_test": X_test_2,
        "Y_train": Y_train_2,
        "Y_test": Y_test_2
    }

    return split1, split2


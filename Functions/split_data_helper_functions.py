from sklearn.model_selection import GroupShuffleSplit

def split_by_recording(df, y, recording_col, test_size=0.2, random_state=42):
    """
    Splits df & y so that rows from the same recording stay together.

    :param df: Feature DataFrame
    :param y: Label Series
    :param recording_col: Column containing the recording ID
    :param test_size: Fraction for test set
    :return: X_train, X_test, y_train, y_test
    """

    groups = df[recording_col]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(splitter.split(df, y, groups))

    X_train = df.iloc[train_idx]
    X_test  = df.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    return X_train, X_test, y_train, y_test
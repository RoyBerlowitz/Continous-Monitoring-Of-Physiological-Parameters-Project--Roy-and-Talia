from xgboost import XGBClassifier

def train_xgboost(X_train, y_train,
                  n_estimators=200,
                  learning_rate=0.1,
                  max_depth=6,
                  subsample=0.8,
                  colsample_bytree=0.8,
                  random_state=42):
    """
    Train a simple XGBoost classifier model.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Training features.
    y_train : array-like
        Training labels.
    n_estimators : int, default=200
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Step size shrinkage.
    max_depth : int, default=6
        Maximum depth of trees.
    subsample : float, default=0.8
        Fraction of samples used per tree.
    colsample_bytree : float, default=0.8
        Fraction of features used per tree.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    model : fitted XGBClassifier model
    """

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    return model

from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, penalty='l2', C=1.0, max_iter=1000):
    """
    Train a simple logistic regression model.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Training features.
    y_train : array-like
        Training labels.
    penalty : str, default='l2'
        Regularization type ('l1', 'l2', 'elasticnet', 'none').
    C : float, default=1.0
        Inverse of regularization strength.
    max_iter : int, default=1000
        Maximum number of iterations.

    Returns
    -------
    model : fitted sklearn LogisticRegression model
    """

    model = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        solver='lbfgs' if penalty == 'l2' else 'saga'
    )

    model.fit(X_train, y_train)
    return model

import sklearn as sk

#diff normalization options
def create_normalizer(method='IQR'):
    if method == 'IQR':
        scaler = sk.preprocessing.RobustScaler(quantile_range=(1., 99.))
    elif method == 'standard':
        scaler = sk.preprocessing.StandardScaler()
    elif method == 'MinMax':
        scaler = sk.preprocessing.MinMaxScaler()
    else:
        raise ValueError("Unknown normalization method")
    return scaler

#build normalization 'func' based on train data and normalize train data
def normalize_fit(data_values, method='IQR'):
    if data_values.ndim == 1:
        data_values = data_values.reshape(-1, 1)

    scaler = create_normalizer(method)
    #updates scaler statistics according to train data and does the transform on train data
    normalized = scaler.fit_transform(data_values)
    return normalized.flatten(), scaler

#takes scaler that was created based on train data and runs normalization in test data
def normalize_transform(data_values, scaler):
    if data_values.ndim == 1:
        data_values = data_values.reshape(-1, 1)

    normalized = scaler.transform(data_values)
    return normalized.flatten()
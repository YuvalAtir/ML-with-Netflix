from sklearn.model_selection import train_test_split


def my_len(x):
    return max(x.shape)


def manual_split(data, n_train):
    data.x_train = data.x[0:n_train, :]
    data.x_validate = data.x[n_train:, :]
    data.y_train = data.y[0:n_train]
    data.y_validate = data.y[n_train:]
    return data


def sklearn_split(data,n_train,seed):
    test_size_percent = n_train / my_len(data.y)
    data.x_train, data.x_validate, data.y_train, data.y_validate = \
        train_test_split(data.x,data.y, test_size=test_size_percent, random_state=seed)
    return data

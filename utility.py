import numpy as np


def count_err(predicted, actual, print_flg):
    num_err = 0
    data_len = len(predicted)
    for obs in range(data_len):
        if np.round(predicted[obs]) != actual[obs]:
            num_err += 1
    if print_flg:
        print('Num of Errors = {}, out of {}, percent = {:0.3f}'.format(num_err, data_len,
                                                                        num_err / data_len))
    return num_err


def srmse(predicted, actual):
    out = np.sqrt(np.mean((predicted - actual) ** 2))
    return out

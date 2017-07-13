import matplotlib.pyplot as plt
import numpy as np
import utility as util
import load_data
from functools import reduce
import operator
import time


def naive_bayes():
    """
    Function: Naive Bayes on Netflix data

    Args: None

    Returns: None

    """
    # read all Netflix data
    data = load_data.Data()

    # append the "intercept", beta0
    #data.x = np.hstack((np.ones((data.n_data, 1)), data.x))

    # data set split: training, cv, test
    split_data_ratio = [0.8, 0, 0.2]
    data.manual_split(split_data_ratio)

    num_movies = 99
    num_rating = 6
    freq_table_xy = np.zeros((num_movies,num_rating,num_rating))   # movies x movies ratings 1-5 x miss c ratings 1-5
    freq_table_y = np.zeros(num_rating)  # miss c ratings 1-5
    freq_table_x = np.zeros((num_movies,num_rating))  # movies ratings 1-5
    for ind_obs in range(data.n_train):
        x = data.x_train[ind_obs, :]
        y = data.y_train[ind_obs]
        freq_table_y[y] += 1
        for indx in range(num_movies):
            freq_table_xy[indx, x[indx], y] += 1
            freq_table_x[indx, x[indx]] += 1

    p_xiy = freq_table_xy / data.n_train
    p_y = freq_table_y / data.n_train
    p_y[0] = 0.0000000001
    p_xi = freq_table_x / data.n_train

    y_est = np.zeros(data.n_test)
    num_err = 0
    for ind_obs in range(data.n_test):
        x = data.x_test[ind_obs, :]
        p_ygx = 1
        #tmp_p_xigy_div_p_x = 1
        for indx in range(num_movies):
            p_ygxi = p_xiy[indx, x[indx], :] / p_xi[indx, x[indx]]   # a vector of all Y for a given x rating for movie xi
            p_ygx *= p_ygxi

        p_ygx = p_ygx/sum(p_ygx)
        y_est_1 = p_ygx[1]*1+p_ygx[2]*2+p_ygx[3]*3+p_ygx[4]*4+p_ygx[5]*5
        y_est[ind_obs] = y_est_1
        #print('The SRMSE for # = {:.3f}'.format(util.srmse(y_est, data.y_test[ind_obs])))

    naive_bayes_srmse = util.srmse(y_est, data.y_test)
    print("Naive Bayes Manual Root Mean squared error: %.3f" % naive_bayes_srmse)
    util.count_err(y_est, data.y_test, 1)


def main():
    start_time = time.time()
    naive_bayes()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__": main()

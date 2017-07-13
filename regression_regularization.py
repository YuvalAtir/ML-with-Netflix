from __future__ import division  # to get a float and not int with simple division /
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import load_data
import utility as util
import time
#import preprocessing


def regularized_regression(reg_type):
    """
    Function: compute regression on Netflix data with regularization

    Args:
        reg_type(int) - 1 = ridge regularization
                        2 = lasso regularization

    Returns: None

    """
    # read all Netflix data
    data = load_data.Data()

    # append the "intercept", beta0
    data.x = np.hstack((np.ones((data.n_data, 1)), data.x))

    # data set split: training, cv, test
    split_data_ratio = [0.8, 0, 0.2]
    data.manual_split(split_data_ratio)

    if reg_type == 1:
        #regr = linear_model.Ridge(alpha=0.01, normalize=False)
        regr = linear_model.RidgeCV(alphas=[0, 1.0, 10, 50, 100, 500, 1000, 3000, 5000], cv=10)
    elif reg_type == 2:
        #regr = linear_model.Lasso(alpha=0.03)
        regr = linear_model.LassoCV(alphas=[0.01, 1.0, 10, 50, 100, 500, 1000, 3000, 5000], cv=10)

    # Train the model using the training sets
    regr.fit(data.x_train, data.y_train)  # calculate coefficients and intercept

    # use model to predict test data
    y_est = regr.predict(data.x_test)
    y_est_srmse = util.srmse(y_est, data.y_test)
    print("Linear Regression With Regularization Using SKLearn Root Mean Squared Error: %.3f" % y_est_srmse)
    util.count_err(y_est, data.y_test, 1)
    # print('Coefficients: \n', regr.coef_)


def find_reg_lamda(reg_type):
    """
        Function: find the hyperparameter lamda for ridge regression on Netflix data

        Args: None

        Returns: None

        """

    if reg_type == 1:
        lamda_vec = (0, 1.0, 10, 50, 100, 500, 1000, 3000, 5000)
    if reg_type == 2:
        lamda_vec = (0.0001, 0.0005, 0.001, 0.003, 0.007, 0.01, 0.05, 0.5, 1)

    alpha_vec_len = len(lamda_vec)

    # read all Netflix data
    data = load_data.Data()

    # append the "intercept", beta0
    data.x = np.hstack((np.ones((data.n_data, 1)), data.x))

    # data set split: training, cv, test
    split_data_ratio = [0.6, 0.2, 0.2]
    data.manual_split(split_data_ratio)

    # time the execution length
    start_time = time.time()

    alpha_vec = [ 1/2*x for x in lamda_vec]
    var_J_train = np.zeros((alpha_vec_len, 1))
    var_J_cv = np.zeros((alpha_vec_len, 1))
    m = 0

    for reg_alpha in alpha_vec:
        print('m = %d' % m)
        if reg_type == 1:
            regr = linear_model.Ridge(alpha=reg_alpha)
        elif reg_type == 2:
            regr = linear_model.Lasso(alpha=reg_alpha)
        regr.fit(data.x_train, data.y_train)
        reg_y_est_train = regr.predict(data.x_train)
        reg_srmse_train = util.srmse(reg_y_est_train, data.y_train)
        var_J_train[m] = reg_srmse_train

        reg_y_est_cv = regr.predict(data.x_validate)
        reg_srmse_cv = util.srmse(reg_y_est_cv, data.y_validate)
        var_J_cv[m] = reg_srmse_cv
        m += 1
    print("--- %s seconds ---" % (time.time() - start_time))

    if reg_type == 1:
        regr = linear_model.Ridge(alpha=3000, normalize=False)
    elif reg_type == 2:
        regr = linear_model.Lasso(alpha=0.05, normalize=False)
    regr.fit(data.x_test, data.y_test)
    reg_y_est_test = regr.predict(data.x_test)
    reg_srmse_test = util.srmse(reg_y_est_test, data.y_test)
    print("Linear Regression With Regularization Manual CV to find Lambda, Root Mean Squared Error: %.3f"
          % reg_srmse_test)
    util.count_err(reg_y_est_test, data.y_test, 1)

    line_tr, = plt.plot(lamda_vec, var_J_train, label='training')
    line_vl, = plt.plot(lamda_vec, var_J_cv, label='validation')
    plt.legend(handles=[line_tr, line_vl])
    plt.xlabel('Ridge Lamda')
    plt.ylabel('RMSE')
    plt.title('Curve for Finding Ridge Lamda')
    plt.show()


def main():
    # reg_type = 1 for ridge, 2 for lasso
    reg_type = 2
    regularized_regression(reg_type)
    find_reg_lamda(reg_type)


if __name__ == "__main__":
    main()

from __future__ import division  # to get a float and not int with simple division /
import numpy as np
from sklearn import linear_model
import random as rand
import load_data
import utility as util
import preprocessing

def linear_regression_sklearn():
    """
    Function: Compute regression on Netflix data using SKLearn's model

    Args: None

    Returns: None

    """
    # read all Netflix data
    data = load_data.Data()

    imp_type = 2
    preprocessing.imputation(data, imp_type)
    data.x = data.x_imputed

    # add the dates data as features
    x_with_dates = preprocessing.add_dates_features(data)
    data.x = x_with_dates

    # data set split: training, cv, test
    split_data_ratio = [0.8, 0, 0.2]
    data.manual_split(split_data_ratio)

    # SKLearn linear regression model init
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(data.x_train, data.y_train)  # calculate coefficients and intercept

    # use model to predict test data
    y_est = regr.predict(data.x_test)
    y_est_srmse = util.srmse(y_est, data.y_test)
    print("Linear Regression Using SKLearn Root Mean Squared Error: %.3f" % y_est_srmse)
    util.count_err(y_est, data.y_test, 1)
    #print('Coefficients: \n', regr.coef_)


def linear_regression_analytical():
    """
    Function: compute regression on Netflix data analytically

    Args: None

    Returns: None

    """
    # read all Netflix data
    data = load_data.Data()

    # append the "intercept", beta0
    data.x = np.hstack((np.ones((data.n_data, 1)), data.x))

    # data set split: training, cv, test
    split_data_ratio = [0.8, 0, 0.2]
    data.manual_split(split_data_ratio)

    beta = np.dot(np.dot(np.linalg.inv(np.dot(data.x_train.transpose(), data.x_train)),
                         data.x_train.transpose()), data.y_train)
    y_est = np.dot(beta.transpose(), data.x_test.transpose())
    y_est_srmse = util.srmse(y_est, data.y_test)
    print("Linear Regression Analytical Root Mean Squared Error: %.3f" % y_est_srmse)
    util.count_err(y_est, data.y_test, 1)
    # print('Coefficients: \n', beta)


def linear_regression_grad_desc():
    """
    Function: compute regression on Netflix data analytically

    Args: None

    Returns: None

    """
    # read all Netflix data
    data = load_data.Data()

    # append the "intercept", beta0
    data.x = np.hstack((np.ones((data.n_data, 1)), data.x))

    # data set split: training, cv, test
    split_data_ratio = [0.8, 0, 0.2]
    data.manual_split(split_data_ratio)

    step = 10
    gd_step_param = 0.001
    tol = 1e-8
    n_features = np.size(data.x_train,1)
    #beta = np.zeros((n_features, 1))
    beta = np.reshape([rand.random()*5 for _ in range(n_features)],(n_features, 1))
    data.y_train = np.reshape(data.y_train, (data.n_train, 1))
    cost_prev = 1000000
    while step > tol:
        y_est = np.dot(data.x_train, beta)
        grad = (np.dot((y_est-data.y_train).transpose(), data.x_train)) / np.size(data.x_train, 0)
        beta -= np.dot(gd_step_param, grad.transpose())
        cost = util.srmse(y_est, data.y_train)
        step = np.absolute(cost - cost_prev)
        cost_prev = cost
        print('cost = {}'.format(cost))
        if cost < 0.7677:
            gd_step_param = 0.0001
    y_est = np.dot(data.x_test, beta)
    y_est_srmse = util.srmse(y_est, data.y_test)
    print("Linear Regression GD Root Mean Squared Error: %.3f" % y_est_srmse)
    util.count_err(y_est, data.y_test, 1)
    # print('Coefficients: \n', beta)


def main():
    funcdict = dict(
        sklearn=linear_regression_sklearn,
        analytical=linear_regression_analytical,
        grad_desc=linear_regression_grad_desc
    )
    funcdict['sklearn']()

if __name__ == "__main__": main()
from __future__ import division  # to get a float and not int with simple division /
import numpy as np
from sklearn import linear_model
import random as rand
import load_data
import utility as util
import preprocessing


def logistic_regression_sklearn():
    """
    Function: Compute logistic regression on Netflix data using SKLearn's model
    use One Vs Rest algorithm
    Regularization determined by parameter C=1/lamda default is 1, no regularization is C=inf
    Regularization showed little effect

    Args: None

    Returns: None

    """
    # read all Netflix data
    data = load_data.Data()

    # data set split: training, cv, test
    split_data_ratio = [0.8, 0, 0.2]
    data.manual_split(split_data_ratio)

    # add the dates data as features
    # x_with_dates = preprocessing.add_dates_features(data)

    # SKLearn linear regression model init
    # Default regularization: C=1/lamda
    logreg = linear_model.LogisticRegression(multi_class='ovr', penalty="l2", C=1)

    # Train the model using the training sets
    logreg.fit(data.x_train, data.y_train)

    # use model to predict test data
    y_est = logreg.predict(data.x_test)

    y_est_srmse = util.srmse(y_est, data.y_test)
    print("Logistic Regression Root Mean Squared Error: %.3f" % y_est_srmse)
    util.count_err(y_est, data.y_test, 1)
    # print('Coefficients: \n', regr.coef_)


def logistic_regression_grad_desc():
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
    tol = 5e-5
    n_features = np.size(data.x_train, 1)
    beta = np.zeros((n_features, 1))
    #beta = np.reshape([rand.random() * 0.5 for _ in range(n_features)], (n_features, 1))
    data.y_train = np.reshape(data.y_train, (data.n_train, 1))
    n_ratings = 6
    beta_ovr = np.zeros((n_ratings, n_features))

    for ovr_ind in range(0, n_ratings):
        y_train_ovr = data.y_train
        y_train_ovr[data.y_train.astype(int) != ovr_ind*np.ones((data.n_train,1))] = 0
        y_train_ovr[data.y_train.astype(int) == ovr_ind*np.ones((data.n_train,1))] = 1

        cost_prev = 1000000
        while step > tol:
            # h = logit / sigmoid function
            h = 1. / (1 + np.exp(-(np.dot(beta.transpose(), data.x_train.transpose()))) )
            # gradient of cost function
            grad = ( np.dot((h - y_train_ovr.transpose()), data.x_train) ) / np.size(data.x_train, 0)
            beta -= np.dot(gd_step_param, grad.transpose())
            # cost function of the logistic regression
            cost = np.mean(-np.dot(y_train_ovr, np.log(h)) - np.dot((1 - y_train_ovr), np.log(1 - h)))
            step = np.absolute(cost - cost_prev)
            cost_prev = cost
            print('cost = {}'.format(cost))
        beta_ovr[ovr_ind, :] = beta.transpose()

    h_ovr = np.zeros((n_ratings, data.n_test))
    for ovr_ind in range(0, n_ratings):
        h_ovr[ovr_ind, :] = 1. / (1 + np.exp(-(np.dot(beta_ovr[ovr_ind, :].transpose(), data.x_test.transpose()))))
    p_y_est = np.amax(h_ovr, axis=0)
    y_est = np.argmax(h_ovr, axis=0)
    util.count_err(y_est, data.y_test, 1)
    y_est_srmse = util.srmse(y_est, data.y_train)
    print("Logistic Regression GD Root Mean Squared Error, OVR = : {}".format(y_est_srmse))
    # print('Coefficients: \n', beta)


def main():
    funcdict = dict(
        sklearn=logistic_regression_sklearn,
        grad_desc=logistic_regression_grad_desc
    )
    funcdict['grad_desc']()

if __name__ == "__main__": main()
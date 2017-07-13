# PCA will cause the X to be in continuous as apposed to discrete range
# which necessitates the use of a float instead of an integer
# PCA does not improve SRMSE but rather reduces the number of coefficients needed
# if we were to reduce X components from the regression, we would reduce the ones that have the smallest P-Value
# meaning they are the least significant, but we don't know anything about cross-correlation between them.
# Because PCA finds n_components orthogonal components it allows us to throw away the small coefficients
# which are NOT correlated to any other components. Without it throwing away some (even small) components
# could have a major influence on other components
# PCA deals only with X, regression deals with X & y so it's best to throw away less significant components in regression after PCA

from __future__ import division  # to get a float and not int with simple division /
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
import load_data
import utility as util


def regression_pca(n_pca):
    """
        Function: Performs PCA transformation for the X Netflix data

        Args:
            n_pca - the number of PCA components (number of X features after PCA feature selection)

        Returns:
            x_pca = the X matrix after PCA transformation

    """
    # read all Netflix data
    data = load_data.Data()

    # append the "intercept", beta0
    data.x = np.hstack((np.ones((data.n_data, 1)), data.x))

    # data set split: training, cv, test
    split_data_ratio = [0.8, 0, 0.2]
    data.manual_split(split_data_ratio)

    pca = PCA(n_components=n_pca)
    pca.fit(data.x_train)
    x_pca = pca.transform(data.x_train)
    # x = pca.inverse_transform(x_pca)

    regr = linear_model.LinearRegression()
    regr.fit(x_pca, data.y_train)
    # The mean squared error
    x_pca_test = pca.transform(data.x_test)
    y_est = regr.predict(x_pca_test)
    y_est_srmse = util.srmse(y_est, data.y_test)
    print("Linear Regression With Regularization Using SKLearn Root Mean Squared Error: %.3f" % y_est_srmse)
    util.count_err(y_est, data.y_test, 1)
    return x_pca


def main():
    # n_pca = the number of PCA components (number of X features after PCA feature selection)
    n_pca = 20
    x_pca = regression_pca(n_pca)


if __name__ == "__main__":
    main()

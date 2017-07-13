import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import load_data
import utility as util


# read all Netflix data
data = load_data.Data()

# data set split: training, cv, test
split_data_ratio = [0.7, 0.3, 0]
data.manual_split(split_data_ratio)

m0 = 100
err_no_vec = np.empty((data.n_train,1))
reg_srmse_vec = np.empty((data.n_train,1))
err_no_vec_trn = np.empty((data.n_train,1))
reg_srmse_vec_trn = np.empty((data.n_train,1))

''' Learning Curve '''
lc_step = 10
for m in range(m0, data.n_train, lc_step):
    print('m = %d' % m)

    ''' MODEL '''
    ridge_alpha = 100
    # regression + regularization to CV
    regr = linear_model.Ridge(alpha=ridge_alpha, normalize=False)
    regr.fit(data.x_train[0:m, :], data.y_train[0:m])

    ''' IN-SAMPLE '''
    reg_y_est_trn = regr.predict(data.x_train[0:m, :])
    reg_srmse_trn = util.srmse(reg_y_est_trn, data.y_train[0:m])
    reg_srmse_vec_trn[m - m0] = reg_srmse_trn
    err_no_trn = util.count_err(reg_y_est_trn, data.y_train[0:m], 0)
    err_no_vec_trn[m - m0] = err_no_trn

    ''' VALIDATION '''
    '''Validation data ends after data.n_validate but the estimator still improves every m'''
    m1 = m
    if m >= data.n_validate:
        m1 = data.n_validate
    reg_y_est = regr.predict(data.x_validate[0:m1])
    reg_srmse = util.srmse(reg_y_est, data.y_validate[0:m1])

    reg_srmse_vec[m - m0] = reg_srmse
    err_no = util.count_err(reg_y_est, data.y_validate, 0)
    err_no_vec[m - m0] = err_no


print("Root Mean squared error: %.3f" % reg_srmse)
plt.plot(range(m0, data.n_train-m0-lc_step, lc_step), reg_srmse_vec_trn[m0:data.n_train-m0-lc_step:lc_step], label='training')
plt.plot(range(m0, data.n_train-m0-lc_step, lc_step), reg_srmse_vec[m0:data.n_train-m0-lc_step:lc_step], label='validation')
plt.xlabel('Size of Training Set')
plt.ylabel('RMSE')
plt.title('Regression Ridge {} = {} Learning Curve'.format(chr(945), ridge_alpha))
plt.show()

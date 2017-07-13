import matplotlib.pyplot as plt
import numpy as np


def plot_est(y_est, y_real=[]):
    plt.plot(y_est, '.')
    if not y_real:
        pass
    else:
        plt.plot(y_real, '.')
    plt.show()

    vec_len = y_real.shape[0]
    plt.plot(y_real.reshape(vec_len, 1) + np.random.rand(vec_len, 1) * 0.5, y_est.reshape(vec_len, 1), '.')
    plt.xlabel('rating - true val')
    plt.ylabel('rating - estimated')
    plt.show()

    print('y_real={}, y_est=[]'.format(y_real, y_est.shape))



    '''
    #######################################################################################################################################################################
    train of 70% CV 10% 10% test 10%+1
    #######################################################################################################################################################################
    '''

    ''' DATA PREPROCESSING '''
    # replace data in X: close corr or mean
    data = Data()
    data_frame = pd.DataFrame(data.x, columns=data.titles)
    correlations = data_frame.corr()
    corr_th = 0.4
    data.replace_zeros_corr(correlations.values, corr_th)
    data.x = data.x_nonzero
    m_train = 8000
    m_validate = 10000 - m_train
    manual_split(data, m_train)

    m0 = 100
    err_no_vec = np.empty((m_train, 1))
    reg_srmse_vec = np.empty((m_train, 1))
    err_no_vec_trn = np.empty((m_train, 1))
    reg_srmse_vec_trn = np.empty((m_train, 1))

    ''' Learning Curve '''
    lc_step = 10
    for m in range(m0, m_train, lc_step):
        print('m = %d' % m)

        ''' MODEL '''
        ridge_alpha = 0
        # regression + regularization to CV
        regr = linear_model.Ridge(alpha=ridge_alpha, normalize=False)
        regr.fit(data.x_train[0:m, :], data.y_train[0:m])
        # print('Coefficients: \n', regr.coef_)

        ''' IN-SAMPLE '''
        reg_y_est_trn = regr.predict(data.x_train[0:m, :])
        reg_srmse_trn = srmse(reg_y_est_trn, data.y_train[0:m])
        reg_srmse_vec_trn[m - m0] = reg_srmse_trn
        err_no_trn = count_err(reg_y_est_trn, data.y_train[0:m], 0)
        err_no_vec_trn[m - m0] = err_no_trn

        ''' VALIDATION '''
        '''Validation data ends after m_validate but the estimator still improves every m'''
        m1 = m
        if m >= m_validate:
            # continue
            m1 = m_validate
        reg_y_est = regr.predict(data.x_validate[0:m1])
        reg_srmse = srmse(reg_y_est, data.y_validate[0:m1])
        # reg_coeff_var = np.sqrt(np.sum((regr.coef_ - np.average(regr.coef_))**2))
        # print("Root Mean squared error: %.3f" % reg_srmse)
        # print('Coefficients Variance score: %.3f' % reg_coeff_var)
        reg_srmse_vec[m - m0] = reg_srmse
        err_no = count_err(reg_y_est, data.y_validate, 0)
        err_no_vec[m - m0] = err_no

    print("Root Mean squared error: %.3f" % reg_srmse)
    plt.plot(range(m0, m_train - m0 - lc_step, lc_step), reg_srmse_vec_trn[m0:m_train - m0 - lc_step:lc_step],
             label='training')
    plt.plot(range(m0, m_train - m0 - lc_step, lc_step), reg_srmse_vec[m0:m_train - m0 - lc_step:lc_step],
             label='validation')
    plt.xlabel('Size of Training Set')
    plt.ylabel('RMSE')
    plt.title('Regression Ridge {} = {} Learning Curve'.format(chr(945), ridge_alpha))
    plt.show()
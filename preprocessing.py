import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def add_dates_features(data):
    # use the dates information as new features
    x_with_dates = np.hstack((data.x, data.dates_x))
    return x_with_dates


def count_zeros(data):
    # count non zeros in data
    zero_count = (data.x == 0).sum()
    zero_count1 = data.x.shape[0]*data.x.shape[1] - np.count_nonzero(data.x, axis=1)
    print(zero_count,zero_count1)
    # count non zeros in data per column
    col_zero_count = (data.x == 0).sum(0)
    plt.plot(col_zero_count, '.')
    plt.xlabel('#movie')
    plt.ylabel('number of missing values')
    plt.title('num of zeros in each column')
    plt.show()  # waits for user to close the window to continue


def imputation(data, imp_type):
    if imp_type == 0:
        data.x_imputed = data.x
    elif imp_type == 1:
        replace_val = np.average(np.average(data.x[data.x != 0]))
        data.x_imputed = data.x
        data.x_imputed[data.x == 0] = replace_val
    elif imp_type == 2:

        # threshold for similarity to "miss congeniality" (test movie)
        corr_th = 0.3

        corr_sp = np.zeros((99, 1))
        corr_np = np.zeros((99, 1))
        pval_sc = np.zeros((99, 1))

        for ind in range(99):
            # correlation between two vectors
            # np.correlate(data.x[:,ind], data.y)  ==  np.dot(data.x[:,ind],data.y)
            corr_np[ind] = np.correlate(data.x[:, ind], data.y)
            # pearsonr returns also the 2-tail p-value
            # scipy.stats.stats.pearsonr(x,y) == np.dot((x-np.mean(x)),(y-np.mean(y)))/(2*np.sqrt(np.var(x))*2*np.sqrt(np.var(y)))
            [coefp, pval] = scipy.stats.stats.pearsonr(data.x[:,ind], data.y)
            corr_sp[ind] = coefp
            pval_sc[ind] = pval
        # corr_sp is used not corr_np (which is brought only for educational purposes)
        plt.plot(range(99), corr_np/max(corr_np), range(99),corr_sp, range(99),pval_sc)
        #plt.show()

        similar_mov = corr_sp[corr_sp > corr_th]
        n_similar = len(similar_mov)
        # the last n_similar are the most similar (highest corr) that match similar_mov
        sorted_corr = np.argsort(corr_sp,axis=0)[-n_similar:]

        # copy the x array and convert integers to floats
        data.x_imputed = data.x[:] * 1.0

        replace_val = np.average(np.average(data.x[data.x != 0]))
        data.x_imputed_tmp = data.x
        data.x_imputed_tmp[data.x == 0] = replace_val

        for mov_idx in range(0, 99):
            users_mov_norating_idx = data.x_imputed[:, mov_idx] == 0
            # first 14 movies have all rating
            if not any(users_mov_norating_idx): continue
            # average of the similar movies where 0 was replaced with mean
            similar_rating = data.x_imputed_tmp[users_mov_norating_idx, sorted_corr]
            # row average for all similar movies of same user
            # replacing zero rating with float will raise num of errors in data but reduce srmse
            data.x_imputed[users_mov_norating_idx, mov_idx] = np.average(similar_rating, axis=0)

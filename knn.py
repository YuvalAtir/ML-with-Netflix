from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import utility as util
import load_data
import time


def knn_sklearn(k):
    """
    Function: Compute K-NN on Netflix data using SKLearn's model
    use One Vs Rest algorithm
    Regularization determined by parameter C=1/lamda default is 1, no regularization is C=inf
    Regularization showed little effect

    Args:
        k(int) - k parameter of K-NN

    Returns: None

    """
    # read all Netflix data
    data = load_data.Data()

    # data set split: training, cv, test
    split_data_ratio = [0.8, 0, 0.2]
    data.manual_split(split_data_ratio)

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(data.x_train, data.y_train)
    y_est = knn.predict(data.x_test)
    knn_srmse = util.srmse(y_est, data.y_test)
    print("K-NN SKLearn Root Mean squared error: %.3f" % knn_srmse)
    util.count_err(y_est, data.y_test, 1)


def knn_manual(k):
    """
        Function: Compute K-NN on Netflix data using SKLearn's model
        use One Vs Rest algorithm
        Regularization determined by parameter C=1/lamda default is 1, no regularization is C=inf
        Regularization showed little effect

        Args:
            k(int) - k parameter of K-NN

        Returns: None

        """
    # read all Netflix data
    data = load_data.Data()

    # data set split: training, cv, test
    split_data_ratio = [0.8, 0, 0.2]
    data.manual_split(split_data_ratio)

    y_est = np.zeros(data.n_test)
    for obs in range(data.n_test):
        x = data.x_test[obs]
        dist = np.linalg.norm((data.x_train-x), ord=2, axis=1)
        sorted_index = np.argsort(dist)
        top = sorted_index[0:k]
        y_est[obs] = np.mean(data.y_train[top])


    knn_srmse = util.srmse(y_est, data.y_test)
    print("K-NN Manual Root Mean squared error: %.3f" % knn_srmse)
    util.count_err(y_est, data.y_test, 1)


def main():
    start_time = time.time()
    k = 20
    funcdict = dict(
        sklearn=knn_sklearn,
        manual=knn_manual
    )
    funcdict['manual'](k)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__": main()
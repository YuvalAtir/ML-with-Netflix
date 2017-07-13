import load_data
from sklearn import discriminant_analysis as da
import utility as util


# read all Netflix data
data = load_data.Data()

# data set split: training, cv, test
split_data_ratio = [0.8, 0, 0.2]
data.manual_split(split_data_ratio)

lda = da.LinearDiscriminantAnalysis()
lda.fit(data.x_train, data.y_train)
y_est_lda = lda.predict(data.x_test)
print('RSMSE of Linear Discrimination Analysis = ', util.srmse(y_est_lda,data.y_test))

qda = da.QuadraticDiscriminantAnalysis(reg_param=0.05)  # regularization barely helps
qda.fit(data.x_train, data.y_train)
y_est_qda = qda.predict(data.x_test)
print('RSMSE of Quadratic Discrimination Analysis = ', util.srmse(y_est_qda,data.y_test))

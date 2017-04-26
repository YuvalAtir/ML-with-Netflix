import matplotlib.pyplot as plt
import numpy as np
from utility import Data
from sklearn import linear_model, model_selection
import main

# regreesion on all rankings (with missing =0!)
# ---------------------------------------------

# read all netflix data
data = Data()
n_train = 8000
main.manual_split(data, n_train)

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(data.x_train, data.y_train)
# The coefficients
print('Coefficients: \n', regr.coef_)   # the _ in coef_ means ^ as in estimate
# The mean squared error
print("Mean squared error: %.2f" % np.mean((regr.predict(data.x_validate) - data.y_validate) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(data.x_validate, data.y_validate))
# Intercept beta0 is not part of Coef
print('intercept = ', regr.intercept_)

# Plot outputs
plt.plot(range(1,2001),regr.predict(data.x_validate),'.')  # same as plt.scatter
plt.plot(range(1,2001),data.y_validate,'.')
plt.show()

#print(data.y_validate.shape,np.random.rand(2000,1).shape,regr.predict(data.x_validate).shape)
plt.plot(data.y_validate.reshape(2000,1)+np.random.rand(2000,1)*0.5,regr.predict(data.x_validate).reshape(2000,1),'.')
plt.xlabel('rating - true val')
plt.ylabel('rating - estimated')
plt.show()


def main():
    pass
'''
In the regular regression there is not enough representation to the outer levels 5 and 2
'''
if __name__ == "__main__": main()
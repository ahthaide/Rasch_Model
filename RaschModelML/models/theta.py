# imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn import datasets, linear_model, metrics
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# load  dataset
from sklearn.externals import joblib

data = pd.read_csv('logitheta.csv')

# defining feature inputs  and outputs
x=data.loc[:,data.columns !="Theta"]
y= data.loc[:,'Theta']
# making sure things are correct



# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5,
                                                    random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error

## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

## plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

## plotting legend
plt.legend(loc='upper right')

## plot title
plt.title("Residual errors")

## function to show plot
# plt.show()

y_pred = reg.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
# Test the model
predictions = reg.predict(X_test)
print(predictions)  # printing predictions

print()  # Printing new line

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
corr, _ = pearsonr(y_test, y_pred)
print('correlation: %.3f' % corr)
accuracy = reg.score(X_test,y_test)
print(accuracy*100,'%')

pkl_filename = "theta_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(reg, file)


def loadmodel(newdata):
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
        newP=pickle_model.predict(newdata)
        print(newP)



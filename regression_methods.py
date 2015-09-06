import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

def evaluatePrediction(azsGT, elevsGT, azsPred, elevsPred):

    errazs = np.arctan2(np.sin(azsGT - azsPred), np.cos(azsGT - azsPred))*180/pi
    errelevs = np.arctan2(np.sin(elevsGT-elevsPred), np.cos(elevsGT-elevsPred))*180/pi
    return errazs, errelevs

def trainRandomForest(xtrain, ytrain):

    rf = RandomForestRegressor()
    r = rf.fit(xtrain, ytrain)

    return rf, r

def testRandomForest(xtest, ytest):

    return rf.predict(X_t)


def trainLinearRegression(xtrain, ytrain):

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(xtrain, ytrain)

    # print('Coefficients: \n', regr.coef_)
    #
    #       % (regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    # Plot outputs
    # plt.scatter(xtest, ytest,  color='black')
    return regr



def testLinearRegression(xtest, ytest):

    return regr.predict(xtest)
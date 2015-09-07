import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

def evaluatePrediction(azsGT, elevsGT, azsPred, elevsPred):

    errazs = np.arctan2(np.sin(azsGT - azsPred), np.cos(azsGT - azsPred))*180/np.pi
    errelevs = np.arctan2(np.sin(elevsGT-elevsPred), np.cos(elevsGT-elevsPred))*180/np.pi
    return errazs, errelevs

def trainRandomForest(xtrain, ytrain):

    randForest = RandomForestRegressor(n_estimators=200, n_jobs=-1)
    rf = randForest.fit(xtrain, ytrain)

    return rf

def testRandomForest(randForest, xtest):

    return randForest.predict(xtest)


def trainLinearRegression(xtrain, ytrain):

    # Create linear regression object
    regr = linear_model.LinearRegression(n_jobs=-1)

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



def testLinearRegression(lrmodel, xtest):

    return lrmodel.predict(xtest)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn import mixture
import ipdb

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


def meanColor(image, win):

    image = np.mean(image[image.shape[0]/2-win:image.shape[0]/2+win,image.shape[1]/2-win:image.shape[1]/2+win,:], axis=0)
    color = np.mean(image, axis=0)

    return color



def colorGMM(image, win):
    np.random.seed(1)
    gmm = mixture.GMM(n_components=8, covariance_type='spherical')
    colors = image[image.shape[0]/2-win:image.shape[0]/2+win,image.shape[1]/2-win:image.shape[1]/2+win,:][:,3]
    gmm.fit(colors)
    gmm._weights=np.array([0.6,0.3,0.1,0,0,0,0,0])
    return gmm

from scipy.stats import vonmises

def poseGMM(azimuth, elevation):
    np.random.seed(1)
    components = [0.7,0.05,0.05,0.05,0.05,0.05,0.05]
    azs = np.random.uniform(0,2*np.pi, 6)
    elevs = np.random.uniform(0,np.pi/2, 6)
    kappa = 50

    vmParamsAz = [(azs[i],kappa) for i in azs]
    vmParamsEl = [(elevs[i], kappa) for i in elevs]

    vmParamsAz = [(azimuth, kappa)]  + vmParamsAz
    vmParamsEl = [(elevation, kappa)]  + vmParamsEl

    return components, vmParamsAz, vmParamsEl


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
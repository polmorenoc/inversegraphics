__author__ = 'pol'
import ipdb
import matplotlib
matplotlib.use('Qt4Agg')
from math import radians
import chumpy as ch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import mixture
from numpy.random import choice

plt.ion()

image = cv2.imread('opendr_GT.png')
image = np.float64(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))/255.0

nComps = 5
nRecComps = 4
gmm = mixture.GMM(n_components=nComps, covariance_type='spherical')
win = 40
colors = image[image.shape[0]/2-win:image.shape[0]/2+win,image.shape[1]/2-win:image.shape[1]/2+win,:].reshape([4*win*win,3])
gmm.fit(colors)

imshape = [win*2,win*2,3]

numPixels = win*2 * win*2
chInput = ch.Ch(colors)
numVars = chInput.size


recSoftmaxW = ch.Ch(np.random.uniform(0,1, [nRecComps,numVars])/numVars)

chRecLogistic = ch.exp(ch.dot(recSoftmaxW,chInput.reshape([numVars,1])))
chRecSoftmax = chRecLogistic.ravel()/ch.sum(chRecLogistic)

chZRecComps = ch.zeros([numVars, nRecComps])

chZ = ch.zeros([numVars])

recMeans = ch.Ch(np.random.uniform(0,1, [3,nRecComps]))
recCovars = 0.2
chRecLogLikelihoods = - 0.5*(chZ.reshape([numPixels,3, 1]) - ch.tile(recMeans, [numPixels, 1,1]))**2 - ch.log((2 * recCovars)  * (1/(ch.sqrt(recCovars) * np.sqrt(2 * np.pi))))

genZCovars = 0.2
chGenComponentsProbs = ch.Ch(gmm.weights_)
chCompMeans = ch.zeros([nComps, 3])

for comp in range(nComps):
    chCompMeans[comp, :] = gmm.means_[comp]

chPZComp = ch.exp( - (ch.tile(chZ.reshape([numPixels,3,1]), [1, 1, nComps]) - chCompMeans.reshape([1,3, nComps]))**2 / (2 * genZCovars))  * (1/(ch.sqrt(genZCovars) * np.sqrt(2 * np.pi)))

chPZ = ch.dot(chGenComponentsProbs.reshape([1,nComps]), chPZComp.reshape([5, numVars]))

prec = 0.5

covars = np.eye(colors.size, colors.size)
# for row in np.arange(covars.shape[0]):
#     cols = [max(row-1, 0), min(row+1,colors.size-1)]
#     covars[row, cols[0]] = prec
#     covars[row, cols[1]] = prec
#     covars[cols[0], row] = prec
#     covars[cols[1], row] = prec

# covars = np.linalg.inv(covars)
detCov = 1
# detCov = np.linalg.det(covars)
# covars = [ch.Ch([covar] for covar in gmm.covars_)]
chResiduals = chInput.ravel() - chZ.ravel()

covar = 0.2

ipdb.set_trace()

chLogJoint = ch.log(chPZ.ravel()) - 0.5*covar*ch.dot(chResiduals,chResiduals)  - 0.5*(ch.log(detCov) + numVars*ch.log((2 * np.pi)))
# chGenMarginal = ch.prod(chLikelihoods)

ipdb.set_trace()

# likelihoodsZ = [chGenComponentsProbs[comp]*ch.exp( - (chInput - chZ)**2 / (2 * covars))  * (1/(ch.sqrt(covars) * np.sqrt(2 * np.pi))) for comp in range(nComps)]
# chLikelihoodsZ = ch.concatenate(likelihoods)
# chGenMarginalZ = ch.exp(ch.sum(ch.log(chLikelihoodsZ)))

gmmRec = mixture.GMM(n_components=nRecComps, covariance_type='spherical')
gmmRec.covars_=gmm.covars_.copy()


#Update the mean of the gaussians and update the mixing weights.
methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead']
free_vars = [recMeans.ravel(), recSoftmaxW]

print("Beginning optimization.")
while True:

    gmmRec.weights_=np.array(chRecSoftmax.r)
    gmmRec.means_=np.array(ch.concatenate(recMeans))
    epsilon = np.random.randn(numVars)
    u = choice(nRecComps, size=1, p=chRecSoftmax.r)
    chZ[:] = chZRecComps[:,u].r.ravel() + recCovars*epsilon.ravel()
    pu = chRecSoftmax
    L = ch.log(pu[u]) + ch.sum(chLogJoint.ravel()) - ch.sum(chRecLogLikelihoods[:,:,u].ravel())
    drL = L.dr_wrt(recMeans)/numPixels
    alpha = 0.1

    recSoftmaxW[:] = recSoftmaxW.r[:] + alpha*L.dr_wrt(recSoftmaxW).reshape(recSoftmaxW.shape)/numPixels
    ipdb.set_trace()
    chZ[:] = chZ.r[:] + alpha*L.dr_wrt(chZ).reshape(chZ.r.shape)/numPixels
    chZRecComps[:,u] = chZ.r[:]
    # ch.minimize({'raw': -L}, bounds=None, method=methods[1], x0=free_vars, callback=None, options={'disp':False, 'maxiter':1})

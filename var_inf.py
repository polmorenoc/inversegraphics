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
nRecComps = 3
gmm = mixture.GMM(n_components=nComps, covariance_type='spherical')
win = 40
colors = image[image.shape[0]/2-win:image.shape[0]/2+win,image.shape[1]/2-win:image.shape[1]/2+win,:].reshape([4*win*win,3])
gmm.fit(colors)

imshape = [win*2,win*2,3]

chInput = ch.Ch(colors)

recSoftmaxW = [ch.Ch([np.random.uniform(0,1, imshape).ravel()])/chInput.size for comp in range(nRecComps)]

recLogistic = [ch.exp(ch.dot(recSoftmaxW[comp],chInput.ravel())) for comp in range(nRecComps)]
chRecLogistic = ch.concatenate(recLogistic)
chRecSoftmax = chRecLogistic/ch.sum(chRecLogistic)

chZ = ch.zeros(imshape)

recMeans = [ch.Ch(np.random.uniform(0,1, 3)) for mean in range(nRecComps)]
recCovars = 0.2
recLikelihoods = [(ch.exp( - (chZ - recMeans[comp])**2 / (2 * recCovars))  * (1/(ch.sqrt(recCovars) * np.sqrt(2 * np.pi)))).ravel() for comp in range(nRecComps)]
chRecLikelihoods = ch.concatenate(recLikelihoods)

chGenComponentsProbs = ch.Ch(gmm.weights_)
means = [ch.Ch(mean) for mean in gmm.means_]
covars = 0.2
# covars = [ch.Ch([covar] for covar in gmm.covars_)]

likelihoods = [chGenComponentsProbs[comp]*ch.exp( - (chInput - means[comp])**2 / (2 * covars))  * (1/(ch.sqrt(covars) * np.sqrt(2 * np.pi))) for comp in range(nComps)]
chLikelihoods = ch.concatenate(likelihoods)
# chGenMarginal = ch.prod(chLikelihoods)

likelihoodsZ = [chGenComponentsProbs[comp]*ch.exp( - (chInput - chZ)**2 / (2 * covars))  * (1/(ch.sqrt(covars) * np.sqrt(2 * np.pi))) for comp in range(nComps)]
chLikelihoodsZ = ch.concatenate(likelihoods)
chGenMarginalZ = ch.exp(ch.sum(ch.log(chLikelihoodsZ)))

gmmRec = mixture.GMM(n_components=nRecComps, covariance_type='spherical')
gmmRec.covars_=gmm.covars_.copy()

chInput = ch.Ch(image.ravel())

#Update the mean of the gaussians and update the mixing weights.
methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead']
free_vars = []
for component in range(nRecComps):
    free_vars = free_vars + [recMeans[component]] + [recSoftmaxW[component]]

print("Beginning optimization.")
while True:
    gmmRec.weights_=np.array(chRecSoftmax.r)
    gmmRec.means_=np.array(ch.concatenate(recMeans))
    epsilon = np.random.randn(3) * 0.04 + 0
    u = choice(nRecComps, size=1, p=chRecSoftmax.r)
    z = np.random.multivariate_normal(recMeans[u],np.eye(3)*covars,[imshape[0],imshape[1]])
    ze = z + epsilon
    chZ[:] = ze
    ipdb.set_trace()
    pu = chRecSoftmax
    L = ch.log(pu) + ch.sum(ch.log(chLikelihoodsZ.ravel())) - ch.sum(ch.log(chRecLikelihoods.ravel()))
    ch.minimize({'raw': -L}, bounds=None, method=methods[1], x0=free_vars, callback=None, options={'disp':False, 'maxiter':1})
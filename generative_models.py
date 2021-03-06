import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import scipy
import chumpy as ch
from chumpy.ch import MatVecMult, Ch, depends_on

def scoreImage(img, template, method, methodParams):
    score = 0

    if method == 'chamferModelToData':
        sqDists = chamferDistanceModelToData(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate'])
        score = np.sum(sqDists)
    elif method == 'robustChamferModelToData':
        sqDists = np.sum(chamferDistanceModelToData(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate']))
        score = robustDistance(sqDists, methodParams['scale'])
    elif method == 'chamferDataToModel':
        sqDists = chamferDistanceDataToModel(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate'])
        score = np.sum(sqDists)
    elif method == 'robustChamferDataToModel':
        sqDists = np.sum(chamferDistanceDataToModel(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate']))
        score = robustDistance(sqDists, methodParams['scale'])
    elif method == 'sqDistImages':
        sqDists = sqDistImages(img, template)
        score = np.sum(sqDists) / template.size
    elif method == 'ignoreSqDistImages':
        sqDists = sqDistImages(img, template)
        score = np.sum(sqDists * (template > 0)) / np.sum(template > 0)
    elif method == 'robustSqDistImages':
        sqDists = sqDistImages(img, template)
        score = robustDistance(sqDists, methodParams['scale'])
    elif method == 'negLogLikelihoodRobust':
        score = -modelLogLikelihoodRobust(img, template,  methodParams['testMask'], methodParams['backgroundModel'], methodParams['layerPrior'], methodParams['variances'])
    elif method == 'negLogLikelihood':
        score = -modelLogLikelihood(img, template,  methodParams['testMask'], methodParams['backgroundModel'], methodParams['variances'])
    return score


def chamferDistanceModelToData(img, template, minThresImage, maxThresImage, minThresTemplate, maxThresTemplate):
    imgEdges = cv2.Canny(np.uint8(img*255), minThresImage,maxThresImage)

    tempEdges = cv2.Canny(np.uint8(template*255), minThresTemplate, maxThresTemplate)

    bwEdges1 = cv2.distanceTransform(~imgEdges, cv2.DIST_L2, 5)

    score = np.sum(np.multiply(tempEdges/255, bwEdges1))/np.sum(tempEdges/255.0)

    return score


def chamferDistanceDataToModel(img, template, minThresImage, maxThresImage, minThresTemplate, maxThresTemplate):
    imgEdges = cv2.Canny(np.uint8(img*255), minThresImage,maxThresImage)

    tempEdges = cv2.Canny(np.uint8(template*255), minThresTemplate, maxThresTemplate)

    bwEdges1 = cv2.distanceTransform(~tempEdges, cv2.DIST_L2, 5)

    score = np.multiply(imgEdges/255.0, bwEdges1)/np.sum(imgEdges/255.0)

    return score


def sqDistImages(img, template):
    sqResiduals = np.square(img - template)
    return sqResiduals


def computeVariances(sqResiduals):
    return np.sum(sqResiduals, axis=3)/sqResiduals.shape[-1]

def pixelLayerPriors(masks):

    return np.sum(masks, axis=2) / masks.shape[-1]

def globalLayerPrior(masks):

    return np.sum(masks) / masks.size

def modelLogLikelihoodRobust(image, template, testMask, backgroundModel, layerPriors, variances):
    likelihood = pixelLikelihoodRobust(image, template, testMask, backgroundModel,  layerPriors, variances)
    liksum = np.sum(np.log(likelihood))


    return liksum

def modelLogLikelihoodRobustCh(image, template, testMask, backgroundModel, layerPriors, variances):
    likelihood = pixelLikelihoodRobustCh(image, template, testMask, backgroundModel,  layerPriors, variances)
    liksum = ch.sum(ch.log(likelihood))

    return liksum

def modelLogLikelihoodRobustRegionCh(image, template, testMask, backgroundModel, layerPriors, variances):
    likelihood = pixelLikelihoodRobustRegionCh(image, template, testMask, backgroundModel,  layerPriors, variances)
    liksum = ch.sum(ch.log(likelihood))

    return liksum

def modelLogLikelihood(image, template, testMask, backgroundModel, variances):
    likelihood = pixelLikelihood(image, template, testMask, backgroundModel, variances)
    liksum = np.sum(np.log(likelihood))

def modelLogLikelihoodCh(image, template, testMask, backgroundModel, variances):
    logLikelihood = logPixelLikelihoodCh(image, template, testMask, backgroundModel, variances)

    return ch.sum(logLikelihood)

def pixelLikelihoodRobust(image, template, testMask, backgroundModel, layerPrior, variances):
    sigma = np.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = np.tile(layerPrior, image.shape[0:2])
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = np.ones(image.shape)

    foregroundProbs = np.prod(1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (image - template)**2 / (2 * variances)) * layerPrior, axis=2) + (1 - repPriors)
    return foregroundProbs * mask + (1-mask)

def pixelLikelihoodRobustSQErrorCh(sqeRenderer, testMask, backgroundModel, layerPrior, variances):
    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(sqeRenderer.r.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, sqeRenderer.r.shape[0:2])
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = np.ones(image.shape)

    probs = ch.exp( - (sqeRenderer) / (2 * variances)) * (1./(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs = (probs[:,:,0] * probs[:,:,1] * probs[:,:,2]) * layerPrior + (1 - repPriors)

    return foregroundProbs * mask + (1-mask)

def pixelLikelihoodRobustCh(image, template, testMask, backgroundModel, layerPrior, variances):
    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, image.shape[0:2])
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = np.ones(image.shape)

    probs = ch.exp( - (image - template)**2 / (2 * variances)) * (1./(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs = (probs[:,:,0] * probs[:,:,1] * probs[:,:,2]) * layerPrior + (1 - repPriors)
    return foregroundProbs * mask + (1-mask)

def pixelLikelihoodRobustRegionCh(image, template, testMask, backgroundModel, layerPrior, variances):
    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, image.shape[0:2])
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = np.ones(image.shape)

    imshape = image.shape
    from opendr.filters import filter_for
    from opendr.filters import GaussianKernel2D

    blur_mtx = filter_for(imshape[0], imshape[1], imshape[2] if len(imshape)>2 else 1, kernel = GaussianKernel2D(3, 1))
    blurred_image = MatVecMult(blur_mtx, image).reshape(imshape)
    blurred_template = MatVecMult(blur_mtx, template).reshape(imshape)

    probs = ch.exp( - (blurred_image - template)**2 / (2 * variances)) * (1./(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs = (probs[:,:,0] * probs[:,:,1] * probs[:,:,2]) * layerPrior + (1 - repPriors)
    return foregroundProbs * mask + (1-mask)



import chumpy as ch
from chumpy import depends_on, Ch

class EdgeFilter(Ch):
    dterms = ['renderer', 'rendererGT']

    def compute_r(self):
        return self.blurredDiff()

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            return self.blurredDiff().dr_wrt(self.renderer)

    def blurredDiff(self):
        edges = self.renderer.boundarybool_image
        imshape = self.renderer.shape
        rgbEdges = np.tile(edges.reshape([imshape[0],imshape[1],1]),[1,1,3]).astype(np.bool)

        from opendr.filters import filter_for
        from opendr.filters import GaussianKernel2D

        blur_mtx = filter_for(imshape[0], imshape[1], imshape[2] if len(imshape)>2 else 1, kernel = GaussianKernel2D(3, 1))
        blurred_diff = MatVecMult(blur_mtx, self.renderer - self.rendererGT).reshape(imshape)

        return blurred_diff[rgbEdges]

class NLLRobustModel(Ch):
    terms = ['Q', 'variances']
    dterms = ['renderer', 'groundtruth']

    def compute_r(self):
        return -np.sum(np.log(self.prob))

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)

            dr = (-1./(self.prob) * fgMask * self.fgProb[:,:,0]*self.fgProb[:,:,1]*self.fgProb[:,:,2] * self.Q[:, :])[:, :, None] * ((self.groundtruth.r - self.renderer.r)/self.variances.r)


            return dr.ravel()

    @depends_on(dterms)
    def fgProb(self):
        return np.exp(- (self.renderer.r - self.groundtruth.r) ** 2 / (2 * self.variances.r)) * (1. / (np.sqrt(self.variances.r) * np.sqrt(2 * np.pi)))

    @depends_on(dterms)
    def prob(self):
        h = self.renderer.r.shape[0]
        w = self.renderer.r.shape[1]

        occProb = np.ones([h, w])
        bgProb = np.ones([h, w])

        fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)

        errorFun = fgMask[:, :]*(self.Q[:, :] * self.fgProb[:,:,0]*self.fgProb[:,:,1]*self.fgProb[:,:,2] + (1-self.Q[:, :]))+ (1- fgMask[:, :])

        return errorFun

    # @depends_on(dterms)
    # def prob(self):
    #     h = self.renderer.r.shape[0]
    #     w = self.renderer.r.shape[1]
    #
    #     occProb = np.ones([h, w])
    #     bgProb = np.ones([h, w])
    #
    #     fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)
    #
    #     errorFun = fgMask[:, :, None] * ((self.Q[0][:, :, None] * self.fgProb) + (self.Q[1] * occProb + self.Q[2] * bgProb)[:, :, None]) + (1 - fgMask[:, :, None])
    #
    #     return errorFun

class NLLRobustSQErrorModel(Ch):
    terms = ['Q', 'variances']
    dterms = ['sqeRenderer']

    def compute_r(self):
        return -np.sum(np.log(self.prob))

    def compute_dr_wrt(self, wrt):

        if wrt is self.sqeRenderer:
            fgMask = np.array(self.sqeRenderer.image_mesh_bool([0])).astype(np.bool)

            # dr = (-1./(self.prob) * fgMask * self.fgProb[:,:,0]*self.fgProb[:,:,1]*self.fgProb[:,:,2] * self.Q[:, :])[:, :, None] * ((self.sqeRenderer.imageGT.r - self.sqeRenderer.render_image)/self.variances.r)
            dr = np.tile((-1./(self.prob) * fgMask * self.fgProb[:,:,0]*self.fgProb[:,:,1]*self.fgProb[:,:,2] * self.Q[:, :])[:, :, None], [1,1,3]) * (-0.5/self.variances.r)

            # dr = (-1./(self.prob[:, :,None]) * fgMask[:, :,None] * self.fgProb * self.Q[:, :,None]) * (-0.5/self.variances.r)


            return dr.ravel()

    @depends_on(dterms)
    def fgProb(self):
        return np.exp(- (self.sqeRenderer.r) / (2 * self.variances.r)) * (1. / (np.sqrt(self.variances.r) * np.sqrt(2 * np.pi)))

    @depends_on(dterms)
    def prob(self):
        h = self.sqeRenderer.r.shape[0]
        w = self.sqeRenderer.r.shape[1]

        occProb = np.ones([h, w])
        bgProb = np.ones([h, w])

        fgMask = np.array(self.sqeRenderer.image_mesh_bool([0])).astype(np.bool)

        errorFun = fgMask[:, :]*(self.Q[:, :] * self.fgProb[:,:,0]*self.fgProb[:,:,1]*self.fgProb[:,:,2] + (1-self.Q[:, :]))+ (1- fgMask[:, :])

        return errorFun

class NLLCRFModel(Ch):
    terms = ['Q', 'variances']
    dterms = ['renderer', 'groundtruth']

    def compute_r(self):
        return -np.sum(np.log(self.prob))

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)

            dr = -1./(self.prob) * fgMask[:, :, None] * self.fgProb * self.Q[0][:, :, None] * ((self.groundtruth.r - self.renderer.r)/self.variances.r)

            return dr.ravel()

    @depends_on(dterms)
    def fgProb(self):
        return np.exp(- (self.renderer.r - self.groundtruth.r) ** 2 / (2 * self.variances.r)) * (1. / (np.sqrt(self.variances.r) * np.sqrt(2 * np.pi)))

    @depends_on(dterms)
    def prob(self):
        h = self.renderer.r.shape[0]
        w = self.renderer.r.shape[1]

        occProb = np.ones([h, w])
        bgProb = np.ones([h, w])

        fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)

        errorFun = fgMask[:, :, None]*((self.Q[0][:, :, None] * self.fgProb) + (self.Q[1] * occProb + self.Q[2] * bgProb)[:, :, None])[:, :, None] + (1- fgMask[:, :, None])

        return errorFun


    @depends_on(dterms)
    def prob(self):
        h = self.renderer.r.shape[0]
        w = self.renderer.r.shape[1]

        occProb = np.ones([h, w])
        bgProb = np.ones([h, w])

        fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)

        errorFun = fgMask[:, :, None] * ((self.Q[0][:, :, None] * self.fgProb) + (self.Q[1] * occProb + self.Q[2] * bgProb)[:, :, None]) + (1 - fgMask[:, :, None])

        return errorFun

class LogCRFModel(Ch):
    terms = ['groundtruth', 'Q', 'variances']
    dterms = ['renderer']

    def compute_r(self):
        return self.logProb()

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            return self.logProb().dr_wrt(self.renderer)

    def logProb(self):

        fgProb = ch.exp(- (self.renderer - self.groundtruth.r) ** 2 / (2 * self.variances.r)) * (1. / (np.sqrt(self.variances.r)* np.sqrt(2 * np.pi)))

        h = self.renderer.r.shape[0]
        w = self.renderer.r.shape[1]

        occProb = np.ones([h, w])
        bgProb = np.ones([h, w])

        fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)

        errorFun = ch.log(fgMask[:, :, None]*((self.Q[0][:, :, None] * fgProb) + (self.Q[1] * occProb + self.Q[2] * bgProb)[:, :, None]) + (1- fgMask[:, :, None]))

        return errorFun


class LogRobustModelOld(Ch):
    dterms = ['renderer', 'groundtruth', 'foregroundPrior', 'variances']

    def compute_r(self):
        return self.logProb()

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            return self.logProb().dr_wrt(self.renderer)

    def logProb(self):
        visibility = self.renderer.visibility_image
        visible = visibility != 4294967295

        visible = np.array(self.renderer.image_mesh_bool([0])).copy().astype(np.bool)

        return ch.log(pixelLikelihoodRobustCh(self.groundtruth, self.renderer, visible, 'MASK', self.foregroundPrior, self.variances))


class LogRobustModel(Ch):
    terms = ['useMask']
    dterms = ['renderer', 'groundtruth', 'foregroundPrior', 'variances']

    def compute_r(self):
        return self.logProb()

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            return self.logProb().dr_wrt(self.renderer)

    def logProb(self):
        # visibility = self.renderer.visibility_image
        # visible = visibility != 4294967295
        try:
            self.useMask
        except:
            self.useMask = False

        if self.useMask:
            visible = self.renderer.indices_image != 0
        else:
            visible = np.ones_like(self.renderer.indices_image.astype(np.bool))

        # visible = np.array(self.renderer.image_mesh_bool([0])).copy().astype(np.bool)

        return ch.log(pixelLikelihoodRobustCh(self.groundtruth, self.renderer, visible, 'MASK', self.foregroundPrior, self.variances))

class LogRobustSQErrorModel(Ch):
    dterms = ['sqeRenderer', 'foregroundPrior', 'variances']

    def compute_r(self):
        return self.logProb()

    def compute_dr_wrt(self, wrt):
        if wrt is self.sqeRenderer:
            return self.logProb().dr_wrt(self.sqeRenderer)

    def logProb(self):
        visibility = self.sqeRenderer.visibility_image
        # visible = visibility != 4294967295

        visible = np.array(self.sqeRenderer.image_mesh_bool([0])).copy().astype(np.bool)

        return ch.log(pixelLikelihoodRobustSQErrorCh(self.sqeRenderer, visible, 'MASK', self.foregroundPrior, self.variances))



class LogRobustModelRegion(Ch):
    dterms = ['renderer', 'groundtruth', 'foregroundPrior', 'variances']

    def compute_r(self):
        return self.logProb()

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            return self.logProb().dr_wrt(self.renderer)

    def logProb(self):
        visibility = self.renderer.visibility_image
        visible = visibility != 4294967295

        visible = np.array(self.renderer.image_mesh_bool([0])).copy().astype(np.bool)

        return ch.log(pixelLikelihoodRobustRegionCh(self.groundtruth, self.renderer, visible, 'MASK', self.foregroundPrior, self.variances))


class LogGaussianModelOld(Ch):
    dterms = ['renderer', 'groundtruth', 'variances']

    def compute_r(self):
        return self.logProb()

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            return self.logProb().dr_wrt(self.renderer)

    def logProb(self):
        visibility = self.renderer.visibility_image
        visible = visibility != 4294967295

        visible = np.array(self.renderer.image_mesh_bool([0])).copy().astype(np.bool)

        return logPixelLikelihoodCh(self.groundtruth, self.renderer, visible, 'MASK', self.variances)

class LogGaussianModel(Ch):
    terms = ['useMask']
    dterms = ['renderer', 'groundtruth', 'variances']

    def compute_r(self):
        return self.logProb()

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            return self.logProb().dr_wrt(self.renderer)

    def logProb(self):
        # visibility = self.renderer.visibility_image
        # visible = visibility != 4294967295
        try:
            self.useMask
        except:
            self.useMask = False

        if self.useMask:
            visible = self.renderer.indices_image != 0 # assumes the first mesh is the background cube.
        else:
            visible = np.ones_like(self.renderer.indices_image.astype(np.bool))

        # visible = np.array(self.renderer.image_mesh_bool([0])).copy().astype(np.bool)

        return logPixelLikelihoodCh(self.groundtruth, self.renderer, visible, 'MASK', self.variances)

def pixelLikelihood(image, template, testMask, backgroundModel, variances):
    sigma = np.sqrt(variances)
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    uniformProbs = np.ones(image.shape[0:2])
    normalProbs = np.prod((1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (image - template)**2 / (2 * variances))),axis=2)
    return normalProbs * mask + (1-mask)


def logPixelLikelihoodCh(image, template, testMask, backgroundModel, variances):
    sigma = ch.sqrt(variances)
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    uniformProbs = np.ones(image.shape[0:2])
    logprobs =   (-(image - template)**2 / (2. * variances))  - ch.log((sigma * np.sqrt(2.0 * np.pi)))
    pixelLogProbs = logprobs[:,:,0] + logprobs[:,:,1] + logprobs[:,:,2]
    return pixelLogProbs * mask


def pixelLikelihoodCh(image, template, testMask, backgroundModel, layerPrior, variances):
    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, image.shape[0:2])
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = np.ones(image.shape)

    probs = ch.exp( - (image - template)**2 / (2 * variances)) * (1./(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs = (probs[:,:,0] * probs[:,:,1] * probs[:,:,2])
    return foregroundProbs * mask + (1-mask)

def layerPosteriorsRobust(image, template, testMask, backgroundModel, layerPrior, variances):

    sigma = np.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = np.tile(layerPrior, image.shape[0:2])
    foregroundProbs = np.prod(1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (image - template)**2 / (2 * variances)) * layerPrior, axis=2)
    backgroundProbs = np.ones(image.shape)
    outlierProbs = (1-repPriors)
    lik = pixelLikelihoodRobust(image, template, testMask, backgroundModel, layerPrior, variances)
    # prodlik = np.prod(lik, axis=2)
    # return np.prod(foregroundProbs*mask, axis=2)/prodlik, np.prod(outlierProbs*mask, axis=2)/prodlik

    return foregroundProbs*mask/lik, outlierProbs*mask/lik

def layerPosteriorsRobustCh(image, template, testMask, backgroundModel, layerPrior, variances):

    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, image.shape[0:2])
    probs = ch.exp( - (image - template)**2 / (2 * variances))  * (1/(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs =  probs[:,:,0] * probs[:,:,1] * probs[:,:,2] * layerPrior
    backgroundProbs = np.ones(image.shape)
    outlierProbs = ch.Ch(1-repPriors)
    lik = pixelLikelihoodRobustCh(image, template, testMask, backgroundModel, layerPrior, variances)
    # prodlik = np.prod(lik, axis=2)
    # return np.prod(foregroundProbs*mask, axis=2)/prodlik, np.prod(outlierProbs*mask, axis=2)/prodlik

    return foregroundProbs*mask/lik, outlierProbs*mask/lik

def layerPosteriorsRobustSQErrorCh(sqeRenderer, testMask, backgroundModel, layerPrior, variances):

    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(sqeRenderer.r.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, sqeRenderer.r.shape[0:2])
    probs = ch.exp( - (sqeRenderer) / (2 * variances))  * (1/(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs =  probs[:,:,0] * probs[:,:,1] * probs[:,:,2] * layerPrior
    backgroundProbs = np.ones(sqeRenderer.r.shape)
    outlierProbs = ch.Ch(1-repPriors)
    lik = pixelLikelihoodRobustSQErrorCh(sqeRenderer, testMask, backgroundModel, layerPrior, variances)
    # prodlik = np.prod(lik, axis=2)
    # return np.prod(foregroundProbs*mask, axis=2)/prodlik, np.prod(outlierProbs*mask, axis=2)/prodlik

    return foregroundProbs*mask/lik, outlierProbs*mask/lik



def robustDistance(sqResiduals, scale):
    return np.sum(sqResiduals/(sqResiduals + scale**2))


def testImageMatching():
    minThresTemplate = 10
    maxThresTemplate = 100
    methodParams = {'scale': 85000, 'minThresImage': minThresTemplate, 'maxThresImage': maxThresTemplate, 'minThresTemplate': minThresTemplate, 'maxThresTemplate': maxThresTemplate}
            
            

    teapots = ["test/teapot1", "test/teapot2","test/teapot3","test/teapot4","test/teapot5","test/teapot6"]

    images = []
    edges = []
    for teapot in teapots:
        im = cv2.imread(teapot + ".png")
        can = cv2.Canny(im, minThresTemplate,maxThresTemplate)
        images.append(im)
        edges.append(can)
        cv2.imwrite(teapot + "_can.png", can)

    confusion = np.zeros([6,6])
    for tp1 in np.arange(1,7):
        for tp2 in np.arange(tp1,7):
            dist = distance = scoreImage(images[tp1-1], images[tp2-1], 'robustSqDistImages', methodParams)
            print(dist)
            confusion[tp1-1, tp2-1] = dist

    plt.matshow(confusion)
    plt.colorbar()
    plt.savefig('test/confusion.png')





# elif method == 'chamferDataToModel':
#         sqDists = chamferDistanceDataToModel(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate'])
#         score = np.sum(sqDists)
#     elif method == 'robustChamferDataToModel':
#         sqDists = np.sum(chamferDistanceDataToModel(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate']))
#         score = robustDistance(sqDists, methodParams['robustScale'])
#     elif method == 'sqDistImages':
#         sqDists = sqDistImages(img, template)
#         score = np.sum(sqDists)
#     elif method == 'robustSqDistImages':
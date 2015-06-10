import cv2
import numpy
import matplotlib.pyplot as plt
import ipdb
import scipy
def scoreImage(img, template, method, methodParams):
    score = 0

    if method == 'chamferModelToData':
        sqDists = chamferDistanceModelToData(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate'])
        score = numpy.sum(sqDists)
    elif method == 'robustChamferModelToData':
        sqDists = numpy.sum(chamferDistanceModelToData(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate']))
        score = robustDistance(sqDists, methodParams['scale'])
    elif method == 'chamferDataToModel':
        sqDists = chamferDistanceDataToModel(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate'])
        score = numpy.sum(sqDists)
    elif method == 'robustChamferDataToModel':
        sqDists = numpy.sum(chamferDistanceDataToModel(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate']))
        score = robustDistance(sqDists, methodParams['scale'])
    elif method == 'sqDistImages':
        sqDists = sqDistImages(img, template)
        score = numpy.sum(sqDists) / template.size
    elif method == 'ignoreSqDistImages':
        sqDists = sqDistImages(img, template)
        score = numpy.sum(sqDists * (template > 0)) / numpy.sum(template > 0)
    elif method == 'robustSqDistImages':
        sqDists = sqDistImages(img, template)
        score = robustDistance(sqDists, methodParams['scale'])
    elif method == 'negLogLikelihoodRobust':
        score = -modelLogLikelihoodRobust(img, template,  methodParams['testMask'], methodParams['backgroundModel'], methodParams['layerPrior'], methodParams['variances'])
    elif method == 'negLogLikelihood':
        score = -modelLogLikelihood(img, template,  methodParams['testMask'], methodParams['backgroundModel'], methodParams['variances'])
    return score


def chamferDistanceModelToData(img, template, minThresImage, maxThresImage, minThresTemplate, maxThresTemplate):
    imgEdges = cv2.Canny(numpy.uint8(img*255), minThresImage,maxThresImage)

    tempEdges = cv2.Canny(numpy.uint8(template*255), minThresTemplate, maxThresTemplate)

    bwEdges1 = cv2.distanceTransform(~imgEdges, cv2.DIST_L2, 5)

    # cv2.imshow('ImageWindow',numpy.uint8(img*255))

    # cv2.waitKey()

    # cv2.imshow('ImageWindow',numpy.uint8(template*255))

    # cv2.waitKey()

    # cv2.imshow('ImageWindow',~tempEdges)

    # cv2.waitKey()

    # ipdb.set_trace()

    # disp = cv2.normalize(bwEdges1, bwEdges1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # cv2.imshow('ImageWindow', disp)

    # cv2.waitKey()

    score = numpy.sum(numpy.multiply(tempEdges/255, bwEdges1))/numpy.sum(tempEdges/255.0)

    return score




def chamferDistanceDataToModel(img, template, minThresImage, maxThresImage, minThresTemplate, maxThresTemplate):
    imgEdges = cv2.Canny(numpy.uint8(img*255), minThresImage,maxThresImage)

    tempEdges = cv2.Canny(numpy.uint8(template*255), minThresTemplate, maxThresTemplate)

    bwEdges1 = cv2.distanceTransform(~tempEdges, cv2.DIST_L2, 5)

    score = numpy.multiply(imgEdges/255.0, bwEdges1)/numpy.sum(imgEdges/255.0)

    return score


def sqDistImages(img, template):
    sqResiduals = numpy.square(img - template)
    return sqResiduals

def computeVariances(sqResiduals):
    return numpy.sum(sqResiduals, axis=3)/sqResiduals.shape[-1]

def pixelLayerPriors(masks):

    return numpy.sum(masks, axis=2) / masks.shape[-1]

def globalLayerPrior(masks):

    return numpy.sum(masks) / masks.size

def modelLogLikelihoodRobust(image, template, testMask, backgroundModel, layerPriors, variances):
    likelihood = pixelLikelihoodRobust(image, template, testMask, backgroundModel,  layerPriors, variances)
    liksum = numpy.sum(numpy.log(likelihood))
    # try:
    #     assert(liksum <= 0)
    # except:
    #     plt.imshow(image)
    #     plt.show()
    #     plt.imshow(template)
    #     plt.show()
    #     ipdb.set_trace()

    return liksum

def modelLogLikelihood(image, template, testMask, backgroundModel, variances):
    likelihood = pixelLikelihood(image, template, testMask, backgroundModel, variances)
    liksum = numpy.sum(numpy.log(likelihood))

    # try:
    #     assert(liksum <= 0)
    # except:
    #     plt.imshow(testMask)
    #     plt.show()
    #     plt.imshow(variances)
    #     plt.show()
    #     plt.imshow(image)
    #     plt.show()
    #     plt.imshow(template)
    #     plt.show()
    #     ipdb.set_trace()

    return liksum

def pixelLikelihoodRobust(image, template, testMask, backgroundModel, layerPrior, variances):
    sigma = numpy.sqrt(variances)
    mask = testMask
    if backgroundModel:
        mask = numpy.ones(image.shape)
    # mask = numpy.repeat(mask[..., numpy.newaxis], 3, 2)
    repPriors = numpy.tile(layerPrior, image.shape[0:2])
    # sum = numpy.sum(numpy.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=numpy.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = numpy.ones(image.shape)

    foregroundProbs = numpy.prod(1/(sigma * numpy.sqrt(2 * numpy.pi)) * numpy.exp( - (image - template)**2 / (2 * variances)) * layerPrior, axis=2) + (1 - repPriors)
    return foregroundProbs * mask + (1-mask)


def pixelLikelihood(image, template, testMask, backgroundModel, variances):
    sigma = numpy.sqrt(variances)
    # sum = numpy.sum(numpy.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=numpy.sqrt(variances) ) + (1 - repPriors)))
    mask = testMask
    if backgroundModel:
        mask = numpy.ones(image.shape[0:2])
    # mask = numpy.repeat(mask[..., numpy.newaxis], 3, 2)
    uniformProbs = numpy.ones(image.shape)
    normalProbs = numpy.prod((1/(sigma * numpy.sqrt(2 * numpy.pi)) * numpy.exp( - (image - template)**2 / (2 * variances))),axis=2)
    return normalProbs * mask + (1-mask)

def layerPosteriorsRobust(image, template, testMask, backgroundModel, layerPrior, variances):

    sigma = numpy.sqrt(variances)
    mask = testMask
    if backgroundModel:
        mask = numpy.ones(image.shape)
    # mask = numpy.repeat(mask[..., numpy.newaxis], 3, 2)
    repPriors = numpy.tile(layerPrior, image.shape[0:2])
    foregroundProbs = numpy.prod(1/(sigma * numpy.sqrt(2 * numpy.pi)) * numpy.exp( - (image - template)**2 / (2 * variances)) * layerPrior, axis=2)
    backgroundProbs = numpy.ones(image.shape)
    outlierProbs = (1-repPriors)
    lik = pixelLikelihoodRobust(image, template, testMask, backgroundModel, layerPrior, variances)
    # prodlik = numpy.prod(lik, axis=2)
    # return numpy.prod(foregroundProbs*mask, axis=2)/prodlik, numpy.prod(outlierProbs*mask, axis=2)/prodlik

    return foregroundProbs*mask/lik, outlierProbs*mask/lik


def robustDistance(sqResiduals, scale):
    return numpy.sum(sqResiduals/(sqResiduals + scale**2))


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

    confusion = numpy.zeros([6,6])
    for tp1 in numpy.arange(1,7):
        for tp2 in numpy.arange(tp1,7):
            dist = distance = scoreImage(images[tp1-1], images[tp2-1], 'robustSqDistImages', methodParams)
            print(dist)
            confusion[tp1-1, tp2-1] = dist

    plt.matshow(confusion)
    plt.colorbar()
    plt.savefig('test/confusion.png')





# elif method == 'chamferDataToModel':
#         sqDists = chamferDistanceDataToModel(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate'])
#         score = numpy.sum(sqDists)
#     elif method == 'robustChamferDataToModel':
#         sqDists = numpy.sum(chamferDistanceDataToModel(img, template, methodParams['minThresImage'], methodParams['maxThresImage'], methodParams['minThresTemplate'],methodParams['maxThresTemplate']))
#         score = robustDistance(sqDists, methodParams['robustScale'])
#     elif method == 'sqDistImages':
#         sqDists = sqDistImages(img, template)
#         score = numpy.sum(sqDists)
#     elif method == 'robustSqDistImages':
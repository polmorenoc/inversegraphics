import numpy as np
import os
import skimage
import skimage.io
import h5py
import ipdb
import scipy.spatial.distance
import image_processing
import matplotlib

__author__ = 'pol'
import recognition_models

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'pdf',
              'axes.labelsize': 10, # fontsize for x and y labels (was 10)
              'axes.titlesize': 10,
              'text.fontsize': 10, # was 10
              'legend.fontsize': 10, # was 10
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

def saveOcclusionPlots(resultDir, occlusions, methodsPred, plotColors, plotMethodsIndices, useShapeModel, meanAbsErrAzsArr, meanAbsErrElevsArr, meanErrorsVColorsCArr, meanErrorsVColorsEArr, meanErrorsLightCoeffsArr, meanErrorsShapeParamsArr, meanErrorsShapeVerticesArr, meanErrorsLightCoeffsCArr, meanErrorsEnvMapArr):

    latexify(columns=2)

    directory = resultDir + 'predictionMeanError-Azimuth'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        ax.plot(occlusions, meanAbsErrAzsArr[method_i], c=plotColors[method_i], label=methodsPred[method_i])
    legend = ax.legend()
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Angular error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'predictionMeanError-Elev'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        ax.plot(occlusions, meanAbsErrElevsArr[method_i], c=plotColors[method_i], label=methodsPred[method_i])
    legend = ax.legend()
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Angular error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'predictionMeanError-VColors-C'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        ax.plot(occlusions, meanErrorsVColorsCArr[method_i], c=plotColors[method_i], label=methodsPred[method_i])
    legend = ax.legend()
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('VColor Error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'predictionMeanError-VColors-E'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        ax.plot(occlusions, meanErrorsVColorsEArr[method_i], c=plotColors[method_i], label=methodsPred[method_i])
    legend = ax.legend()
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Vertex Color error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'predictionMeanError-SH'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        ax.plot(occlusions, meanErrorsLightCoeffsArr[method_i], c=plotColors[method_i], label=methodsPred[method_i])
    legend = ax.legend()
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Mean SH coefficients error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    if useShapeModel:
        directory = resultDir + 'predictionMeanError-ShapeParams'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for method_i in plotMethodsIndices:
            ax.plot(occlusions, meanErrorsShapeParamsArr[method_i], c=plotColors[method_i], label=methodsPred[method_i])
        legend = ax.legend()
        ax.set_xlabel('Occlusion (\%)')
        ax.set_ylabel('Mean Shape Parameters error')

        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_xlim((0, 100))
        ax.set_ylim((-0.0, y2))
        ax.set_title('Cumulative prediction per occlusion level')
        fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
        plt.close(fig)

        directory = resultDir + 'predictionMeanError-ShapeVertices'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for method_i in plotMethodsIndices:
            ax.plot(occlusions, meanErrorsShapeVerticesArr[method_i], c=plotColors[method_i], label=methodsPred[method_i])
        legend = ax.legend()
        ax.set_xlabel('Occlusion (\%)')
        ax.set_ylabel('Shape vertices error')

        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_xlim((0, 100))
        ax.set_ylim((-0.0, y2))
        ax.set_title('Cumulative prediction per occlusion level')
        fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
        plt.close(fig)

    directory = resultDir + 'predictionMeanError-SH-C'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        ax.plot(occlusions, meanErrorsLightCoeffsCArr[method_i], c=plotColors[method_i], label=methodsPred[method_i])
    legend = ax.legend()
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('SH coefficients error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'predictionMeanError-SH-EnvMap'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        ax.plot(occlusions, meanErrorsEnvMapArr[method_i], c=plotColors[method_i], label=methodsPred[method_i])
    legend = ax.legend()
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('SH Environment Map error change')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

from numpy.core.umath_tests import matrix_multiply
def scaleInvariantMSECoeff(x_pred, x_target):
    #Rows: test samples
    #Cols: target variables
    scales = (matrix_multiply(x_pred[:,None,:],x_target[:,:,None])/matrix_multiply(x_pred[:,None,:],x_pred[:,:,None])).ravel()

    return scales


def euclidean(X,Y):
    """
    x: matrix (N,D), each row is an datapoint
    y: matrix (M,D), each row is an datapoint
    A MxN matrix of Euclidean distances is returned
    """
    return scipy.spatial.distance.cdist(X, Y[None,:], 'euclidean').T

def one_nn(x_train, x_test, distance_f=euclidean):
    """
    """
    distances = distance_f(x_train, x_test)
    nn_indices = np.argmin(distances, 1)
    return nn_indices


def shapeVertexErrors(chShapeParams, chVertices, testShapeParamsGT, shapeParamsPred):
    oldShapeParams = chShapeParams.r.copy()

    errorsShapeVertices = np.array([])
    for test_i in range(len(testShapeParamsGT)):

        chShapeParams[:] = testShapeParamsGT[test_i].copy()
        vertsGT = chVertices.r.copy()
        chShapeParams[:] = shapeParamsPred[test_i].copy()
        vertsPred = chVertices.r.copy()
        errorShapeVertices = np.sqrt(np.sum((vertsPred - vertsGT)**2))
        errorsShapeVertices = np.append(errorsShapeVertices,errorShapeVertices)

    chShapeParams[:] = oldShapeParams
    return errorsShapeVertices

def computeErrors(setTest, azimuths, testAzsRel, elevations, testElevsGT, vColors, testVColorGT, lightCoeffs, testLightCoefficientsGTRel, approxProjections,  approxProjectionsGT, shapeParams, testShapeParamsGT, useShapeModel=False, chShapeParams=None, chVertices=None):

    errorsPosePredList = []
    errorsLightCoeffsList = []
    errorsShapeParamsList = []
    errorsShapeVerticesList = []
    errorsEnvMapList = []
    errorsLightCoeffsCList = []
    errorsVColorsEList = []
    errorsVColorsCList = []

    for method in range(len(azimuths)):

        azsPred = azimuths[method]
        elevsPred = elevations[method]
        vColorsPred = vColors[method]
        relLightCoefficientsPred = lightCoeffs[method]
        approxProjectionsPred = approxProjections[method]

        if useShapeModel:
            shapeParamsPred = shapeParams[method]

        errorsPosePred = recognition_models.evaluatePrediction(testAzsRel[setTest], testElevsGT[setTest], azsPred[setTest], elevsPred[setTest])
        errorsPosePredList = errorsPosePredList + [errorsPosePred]

        errorsLightCoeffs = (testLightCoefficientsGTRel[setTest] - relLightCoefficientsPred[setTest]) ** 2
        errorsLightCoeffsList = errorsLightCoeffsList + [errorsLightCoeffs]

        if useShapeModel:
            errorsShapeParams = (testShapeParamsGT[setTest] - shapeParamsPred[setTest]) ** 2
            errorsShapeParamsList = errorsShapeParamsList + [errorsShapeParams]

            errorsShapeVertices = shapeVertexErrors(chShapeParams, chVertices, testShapeParamsGT[setTest], shapeParamsPred[setTest])
            errorsShapeVerticesList = errorsShapeVerticesList + [errorsShapeVertices]


        envMapProjScaling = scaleInvariantMSECoeff(approxProjectionsPred.reshape([len(approxProjectionsPred), -1])[setTest], approxProjectionsGT.reshape([len(approxProjectionsGT), -1])[setTest])
        errorsEnvMap = (approxProjectionsGT[setTest] -  envMapProjScaling[:,None, None]*approxProjectionsPred[setTest])**2
        errorsEnvMapList=  errorsEnvMapList + [errorsEnvMap]

        envMapScaling = scaleInvariantMSECoeff(relLightCoefficientsPred[setTest], testLightCoefficientsGTRel[setTest])
        errorsLightCoeffsC = (testLightCoefficientsGTRel[setTest] - envMapScaling[:,None]* relLightCoefficientsPred[setTest]) ** 2
        errorsLightCoeffsCList = errorsLightCoeffsCList + [errorsLightCoeffsC]

        errorsVColorsE = image_processing.eColourDifference(testVColorGT[setTest], vColorsPred[setTest])
        errorsVColorsEList = errorsVColorsEList + [errorsVColorsE]

        errorsVColorsC = image_processing.cColourDifference(testVColorGT[setTest], vColorsPred[setTest])
        errorsVColorsCList = errorsVColorsCList + [errorsVColorsC]

    return errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList


def computeErrorMeans(testSet, useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList):
    meanAbsErrAzsList = []
    meanAbsErrElevsList = []
    medianAbsErrAzsList = []
    medianAbsErrElevsList = []
    meanErrorsLightCoeffsList = []
    meanErrorsShapeParamsList = []
    meanErrorsShapeVerticesList = []
    meanErrorsLightCoeffsCList = []
    meanErrorsEnvMapList = []
    meanErrorsVColorsEList = []
    meanErrorsVColorsCList = []

    for method_i in range(len(errorsPosePredList)):
        meanAbsErrAzsList = meanAbsErrAzsList + [np.mean(np.abs(errorsPosePredList[method_i][0][testSet]))]
        meanAbsErrElevsList = meanAbsErrElevsList + [np.mean(np.abs(errorsPosePredList[method_i][1][testSet]))]

        medianAbsErrAzsList = medianAbsErrAzsList + [np.median(np.abs(errorsPosePredList[method_i][0][testSet]))]
        medianAbsErrElevsList = medianAbsErrElevsList + [np.median(np.abs(errorsPosePredList[method_i][1][testSet]))]

        meanErrorsLightCoeffsList = meanErrorsLightCoeffsList + [np.mean(np.mean(errorsLightCoeffsList[method_i][testSet],axis=1), axis=0)]

        if useShapeModel:
            meanErrorsShapeParamsList = meanErrorsShapeParamsList + [np.mean(np.mean(errorsShapeParamsList[method_i][testSet],axis=1), axis=0)]
            meanErrorsShapeVerticesList = meanErrorsShapeVerticesList + [np.mean(errorsShapeVerticesList[method_i][testSet], axis=0)]

        meanErrorsLightCoeffsCList = meanErrorsLightCoeffsCList + [np.mean(np.mean(errorsLightCoeffsCList[method_i][testSet],axis=1), axis=0)]
        meanErrorsEnvMapList = meanErrorsEnvMapList + [np.mean(errorsEnvMapList[method_i][testSet])]
        meanErrorsVColorsEList = meanErrorsVColorsEList + [np.mean(errorsVColorsEList[method_i][testSet], axis=0)]
        meanErrorsVColorsCList = meanErrorsVColorsCList + [np.mean(errorsVColorsCList[method_i][testSet], axis=0)]

    return meanAbsErrAzsList, meanAbsErrElevsList, medianAbsErrAzsList, medianAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList


def writeImagesHdf5(imagesDir, writeDir, imageSet, writeGray=False ):
    print("Writing HDF5 file")
    image = skimage.io.imread(imagesDir + 'im' + str(imageSet[0]) + '.jpeg')
    imDtype = image.dtype
    width = image.shape[1]
    height = image.shape[0]
    if not writeGray:
        gtDataFile = h5py.File(writeDir + 'images.h5', 'w')
        images = np.array([], dtype = np.dtype('('+ str(height)+','+ str(width) +',3)uint8'))
        gtDataset = gtDataFile.create_dataset("images", data=images, maxshape=(None,height,width, 3))
        # images = np.zeros([len(imageSet), height, width, 3], dtype=np.uint8)
    else:
        imageGray = 0.3*image[:,:,0] + 0.59*image[:,:,1] + 0.11*image[:,:,2]
        grayDtype = imageGray.dtype

        gtDataFile = h5py.File(writeDir + 'images_gray.h5', 'w')
        # images = np.zeros([], dtype=np.float32)
        images = np.array([], dtype = np.dtype('('+ str(height)+','+ str(width) +')f'))
        gtDataset = gtDataFile.create_dataset("images", data=images, maxshape=(None,height,width))

    for imageit, imageid  in enumerate(imageSet):
        gtDataset.resize(gtDataset.shape[0]+1, axis=0)

        image = skimage.io.imread(imagesDir + 'im' + str(imageid) + '.jpeg').astype(np.uint8)
        if not writeGray:
            gtDataset[-1] = image
        else:
            image = image.astype(np.float32)/255.0
            gtDataset[-1] = 0.3*image[:,:,0] + 0.59*image[:,:,1] + 0.11*image[:,:,2]

        gtDataFile.flush()

    gtDataFile.close()
    print("Ended writing HDF5 file")

def loadMasks(imagesDir, maskSet):
    masks = []
    for imageit, imageid  in enumerate(maskSet):

        if os.path.isfile(imagesDir + 'mask' + str(imageid) + '.npy'):
            masks = masks + [np.load(imagesDir + 'mask' + str(imageid) + '.npy')[None,:,:]]

    return np.vstack(masks)

def readImages(imagesDir, imageSet, loadGray=False, loadFromHdf5=False):
    if loadFromHdf5:
        if not loadGray:
            if os.path.isfile(imagesDir + 'images.h5'):
                gtDataFile = h5py.File(imagesDir + 'images.h5', 'r')
                boolSet = np.zeros(gtDataFile["images"].shape[0]).astype(np.bool)
                boolSet[imageSet] = True
                return gtDataFile["images"][boolSet,:,:,:].astype(np.float32)/255.0
        else:
            if os.path.isfile(imagesDir + 'images_gray.h5'):
                gtDataFile = h5py.File(imagesDir + 'images_gray.h5', 'r')
                boolSet = np.zeros(gtDataFile["images"].shape[0]).astype(np.bool)
                boolSet[imageSet] = True
                return gtDataFile["images"][boolSet,:,:].astype(np.float32)
    else:
        image = skimage.io.imread(imagesDir + 'im' + str(imageSet[0]) + '.jpeg')
        width = image.shape[1]
        height = image.shape[0]
        if not loadGray:
            images = np.zeros([len(imageSet), height, width, 3], dtype=np.float32)
        else:
            images = np.zeros([len(imageSet), height, width], dtype=np.float32)
        for imageit, imageid  in enumerate(imageSet):
            if os.path.isfile(imagesDir + 'im' + str(imageid) + '.jpeg'):
                image = skimage.io.imread(imagesDir + 'im' + str(imageid) + '.jpeg')
            else:
                print("Image " + str(imageid) + " does not exist!")
                image = np.zeros_like(image)
            image = image/255.0
            if not loadGray:
                images[imageit, :, :, :] =  image
            else:
                images[imageit, :, :] =  0.3*image[:,:,0] + 0.59*image[:,:,1] + 0.11*image[:,:,2]

        return images

def readImagesHdf5(imagesDir, loadGray=False):
    if not loadGray:
        if os.path.isfile(imagesDir + 'images.h5'):
             gtDataFile = h5py.File(imagesDir + 'images.h5', 'r')
             return gtDataFile["images"]
        else:
            if os.path.isfile(imagesDir + 'images_gray.h5'):
                gtDataFile = h5py.File(imagesDir + 'images_gray.h5', 'r')
                return gtDataFile["images"]

def generateExperiment(size, experimentDir, ratio, seed):
    np.random.seed(seed)
    data = np.arange(size)
    np.random.shuffle(data)
    train = data[0:np.int(size*ratio)]
    test = data[np.int(size*ratio)::]

    if not os.path.exists(experimentDir):
        os.makedirs(experimentDir)

    np.save(experimentDir + 'train.npy', train)
    np.save(experimentDir + 'test.npy', test)

# saveScatter(xaxis*180/np.pi, yaxis[1], 'Azimuth error (ground-truth)', Azimuth (predicted), filename)

import matplotlib.pyplot as plt
def saveScatter(xaxis, yaxis, xlabel, ylabel, filename):
    plt.ioff()
    fig = plt.figure()
    plt.scatter(xaxis, yaxis)
    plt.xlabel('Elevation (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,90,-90,90))
    plt.title('Performance scatter plot')
    fig.savefig(filename)
    plt.close(fig)

#Method from https://github.com/adamlwgriffiths/Pyrr/blob/master/pyrr/geometry.py
def create_cube(scale=(1.0,1.0,1.0), st=False, rgba=np.array([1.,1.,1.,1.]), dtype='float32', type='triangles'):
    """Returns a Cube reading for rendering."""

    shape = [24, 3]
    rgba_offset = 3

    width, height, depth = scale
    # half the dimensions
    width /= 2.0
    height /= 2.0
    depth /= 2.0

    vertices = np.array([
        # front
        # top right
        ( width, height, depth,),
        # top left
        (-width, height, depth,),
        # bottom left
        (-width,-height, depth,),
        # bottom right
        ( width,-height, depth,),

        # right
        # top right
        ( width, height,-depth),
        # top left
        ( width, height, depth),
        # bottom left
        ( width,-height, depth),
        # bottom right
        ( width,-height,-depth),

        # back
        # top right
        (-width, height,-depth),
        # top left
        ( width, height,-depth),
        # bottom left
        ( width,-height,-depth),
        # bottom right
        (-width,-height,-depth),

        # left
        # top right
        (-width, height, depth),
        # top left
        (-width, height,-depth),
        # bottom left
        (-width,-height,-depth),
        # bottom right
        (-width,-height, depth),

        # top
        # top right
        ( width, height,-depth),
        # top left
        (-width, height,-depth),
        # bottom left
        (-width, height, depth),
        # bottom right
        ( width, height, depth),

        # bottom
        # top right
        ( width,-height, depth),
        # top left
        (-width,-height, depth),
        # bottom left
        (-width,-height,-depth),
        # bottom right
        ( width,-height,-depth),
    ], dtype=dtype)

    st_values = None
    rgba_values = None

    if st:
        # default st values
        st_values = np.tile(
            np.array([
                (1.0, 1.0,),
                (0.0, 1.0,),
                (0.0, 0.0,),
                (1.0, 0.0,),
            ], dtype=dtype),
            (6,1,)
        )

        if isinstance(st, bool):
            pass
        elif isinstance(st, (int, float)):
            st_values *= st
        elif isinstance(st, (list, tuple, np.ndarray)):
            st = np.array(st, dtype=dtype)
            if st.shape == (2,2,):
                # min / max
                st_values *= st[1] - st[0]
                st_values += st[0]
            elif st.shape == (4,2,):
                # per face st values specified manually
                st_values[:] = np.tile(st, (6,1,))
            elif st.shape == (6,2,):
                # st values specified manually
                st_values[:] = st
            else:
                raise ValueError('Invalid shape for st')
        else:
            raise ValueError('Invalid value for st')

        shape[-1] += st_values.shape[-1]
        rgba_offset += st_values.shape[-1]

    if len(rgba) > 0:
        # default rgba values
        rgba_values = np.tile(np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype), (24,1,))

        if isinstance(rgba, bool):
            pass
        elif isinstance(rgba, (int, float)):
            # int / float expands to RGBA with all values == value
            rgba_values *= rgba
        elif isinstance(rgba, (list, tuple, np.ndarray)):
            rgba = np.array(rgba, dtype=dtype)

            if rgba.shape == (3,):
                rgba_values = np.tile(rgba, (24,1,))
            elif rgba.shape == (4,):
                rgba_values[:] = np.tile(rgba, (24,1,))
            elif rgba.shape == (4,3,):
                rgba_values = np.tile(rgba, (6,1,))
            elif rgba.shape == (4,4,):
                rgba_values = np.tile(rgba, (6,1,))
            elif rgba.shape == (6,3,):
                rgba_values = np.repeat(rgba, 4, axis=0)
            elif rgba.shape == (6,4,):
                rgba_values = np.repeat(rgba, 4, axis=0)
            elif rgba.shape == (24,3,):
                rgba_values = rgba
            elif rgba.shape == (24,4,):
                rgba_values = rgba
            else:
                raise ValueError('Invalid shape for rgba')
        else:
            raise ValueError('Invalid value for rgba')

        shape[-1] += rgba_values.shape[-1]

    data = np.empty(shape, dtype=dtype)
    data[:,:3] = vertices
    if st_values is not None:
        data[:,3:5] = st_values
    if rgba_values is not None:
        data[:,rgba_offset:] = rgba_values

    if type == 'triangles':
        # counter clockwise
        # top right -> top left -> bottom left
        # top right -> bottom left -> bottom right
        indices = np.tile(np.array([0, 1, 2, 0, 2, 3], dtype='int'), (6,1))
        for face in range(6):
            indices[face] += (face * 4)
        indices.shape = (-1,)


    return data, indices
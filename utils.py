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
import pickle

def joinExperiments(range1, range2, testSet1,methodsPred1,testOcclusions1,testPrefixBase1,parameterRecognitionModels1,azimuths1,elevations1,vColors1,lightCoeffs1,likelihoods1,shapeParams1,segmentations1, approxProjections1, approxProjectionsGT1, testSet2,methodsPred2,testOcclusions2,testPrefixBase2,parameterRecognitionModels2,azimuths2,elevations2,vColors2,lightCoeffs2,likelihoods2,shapeParams2,segmentations2, approxProjections2, approxProjectionsGT2):

    testSet = np.append(testSet1, testSet2)
    parameterRecognitionModels = [parameterRecognitionModels1] + [parameterRecognitionModels2]
    testPrefixBase = [testPrefixBase1] + [testPrefixBase2]
    methodsPred = methodsPred1
    testOcclusions = np.append(testOcclusions1, testOcclusions2)

    azimuths = []
    elevations = []
    vColors = []
    lightCoeffs = []
    shapeParams = []
    likelihoods = []
    segmentations = []
    approxProjections = []
    approxProjectionsGT = []

    for method in range(len(methodsPred)):
        if azimuths1[method] is not None and azimuths2[method] is not None:
            azimuths  = azimuths + [np.append(azimuths1[method][range1], azimuths2[method][range2])]
        else:
            azimuths  = azimuths + [None]
        if elevations1[method] is not None and elevations2[method] is not None:
            elevations = elevations + [np.append(elevations1[method][range1], elevations2[method][range2])]
        else:
            elevations = elevations + [None]
        if vColors1[method] is not None and vColors2[method] is not None:
            vColors = vColors + [np.vstack([vColors1[method][range1], vColors2[method][range2]])]
        else:
            vColors = vColors + [None]
        if lightCoeffs1[method] is not None and lightCoeffs2[method] is not None:
            lightCoeffs = lightCoeffs + [np.vstack([lightCoeffs1[method][range1], lightCoeffs2[method][range2]])]
        else:
            lightCoeffs = lightCoeffs + [None]
        if shapeParams1[method] is not None and shapeParams2[method] is not None:
            shapeParams = shapeParams + [np.vstack([shapeParams1[method][range1], shapeParams2[method][range2]])]
        else:
            shapeParams = shapeParams + [None]
        if likelihoods1[method] is not None and likelihoods2[method] is not None:
            likelihoods = likelihoods + [np.append(likelihoods1[method][range1], likelihoods2[method][range2])]
        else:
            likelihoods = likelihoods + [None]

        if segmentations1[method] is not None and segmentations2[method] is not None:
            segmentations = segmentations + [np.vstack([segmentations1[method][range1], segmentations2[method][range2]])]
        else:
            segmentations = segmentations + [None]

        if approxProjections1 is not None and approxProjections2 is not None:
            if approxProjections1[method] is not None and approxProjections2[method] is not None:
                approxProjections = approxProjections + [np.vstack([approxProjections1[method][range1], approxProjections2[method][range2]])]
            else:
                approxProjections = approxProjections + [None]

        if approxProjectionsGT1 is not None and approxProjectionsGT2 is not None:
            if approxProjectionsGT1[method] is not None and approxProjectionsGT2[method] is not None:
                approxProjectionsGT = approxProjectionsGT + [np.vstack([approxProjectionsGT1[method][range1], approxProjectionsGT2[method][range2]])]
            else:
                approxProjectionsGT = approxProjectionsGT + [None]

    return testSet, parameterRecognitionModels, testPrefixBase, methodsPred, testOcclusions, azimuths, elevations, vColors, lightCoeffs, shapeParams, likelihoods, segmentations, approxProjections, approxProjectionsGT

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figue.

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

    if fig_width is None:        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'pdf',
              'axes.labelsize': 12, # fontsize for x and y labels (was 10)
              'axes.titlesize': 12,
              'font.size': 12, # was 10
              'legend.fontsize': 12, # was 10
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

def saveScatterPlots(resultDir, testOcclusions, useShapeModel, errorsPosePred, errorsPoseFitted,errorsLightCoeffsC,errorsFittedLightCoeffsC,errorsEnvMap,errorsFittedEnvMap,errorsLightCoeffs,errorsFittedLightCoeffs,errorsShapeParams,errorsFittedShapeParams,errorsShapeVertices,errorsFittedShapeVertices,errorsVColorsE,errorsFittedVColorsE,errorsVColorsC,errorsFittedVColorsC, errorsVColorsS,errorsFittedVColorsS):

    latexify(columns=2)

    directory = resultDir + 'pred-azimuth-errors_fitted-azimuth-error'
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(abs(errorsPosePred[0]), abs(errorsPoseFitted[0]), s=20, vmin=0, vmax=100,
                      c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted azimuth errors')
    ax.set_ylabel('Fitted azimuth errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    for test_i in range(len(errorsPosePred[0])):
        if abs(errorsPoseFitted[0][test_i]) > abs(errorsPosePred[0][test_i]) + 10:
            plt.annotate(
                str(test_i),
                xy=(abs(errorsPosePred[0][test_i]), abs(errorsPoseFitted[0][test_i])), xytext=(0, 15),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    ax.set_title('Recognition vs fitted azimuth errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'pred-elevation-errors_fitted-elevation-error'
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(abs(errorsPosePred[1]), abs(errorsPoseFitted[1]), s=20, vmin=0, vmax=100,
                      c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
    for test_i in range(len(errorsPosePred[1])):
        if abs(errorsPoseFitted[1][test_i]) > abs(errorsPosePred[1][test_i]) + 10:
            plt.annotate(
                str(test_i),
                xy=(abs(errorsPosePred[1][test_i]), abs(errorsPoseFitted[1][test_i])), xytext=(15, 15),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),

    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted elevation errors')
    ax.set_ylabel('Fitted elevation errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted azimuth errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'shcoeffsserrorsC_fitted-shcoeffsserrorsC'

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(np.mean(errorsLightCoeffsC, axis=1), np.mean(errorsFittedLightCoeffsC, axis=1), s=20,
                      vmin=0, vmax=100, c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
    for test_i in range(len(errorsLightCoeffsC)):
        if np.mean(errorsFittedLightCoeffsC, axis=1)[test_i] > np.mean(errorsLightCoeffsC, axis=1)[test_i] + 0.1:
            plt.annotate(
                str(test_i),
                xy=(np.mean(errorsLightCoeffsC, axis=1)[test_i], np.mean(errorsFittedLightCoeffsC, axis=1)[test_i]), xytext=(0, 15),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted SH coefficients errors')
    ax.set_ylabel('Fitted SH coefficients errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted SH coefficients errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    if errorsEnvMap is not None and  errorsFittedEnvMap is not None:
        directory = resultDir + 'shcoeffsserrorsC_fitted-shEnvMap'
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(errorsEnvMap, errorsFittedEnvMap, s=20,
                          vmin=0, vmax=100, c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted SH coefficients errors')
        ax.set_ylabel('Fitted SH coefficients errors')
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_xlim((0, max(x2,y2)))
        ax.set_ylim((0, max(x2,y2)))
        ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
        ax.set_title('Recognition vs fitted SH Environment map errors')
        fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
        plt.close(fig)

    directory = resultDir + 'shcoeffsserrors_fitted-shcoeffsserrors'
    # Show scatter correlations with occlusions.
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(np.mean(errorsLightCoeffs, axis=1), np.mean(errorsFittedLightCoeffs, axis=1), s=20,
                      vmin=0, vmax=100, c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted SH coefficients errors')
    ax.set_ylabel('Fitted SH coefficients errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted SH coefficients errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    if useShapeModel:
        directory = resultDir + 'shapeparamserrors_fitted-shapeparamserrors'
        # Show scatter correlations with occlusions.
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(np.mean(errorsShapeParams, axis=1), np.mean(errorsFittedShapeParams, axis=1), s=20,
                          vmin=0, vmax=100, c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted shape parameters errors')
        ax.set_ylabel('Fitted shape parameters errors')
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_xlim((0, max(x2,y2)))
        ax.set_ylim((0, max(x2,y2)))
        ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
        ax.set_title('Recognition vs fitted Shape parameters errors')
        fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
        plt.close(fig)

        if errorsShapeVertices is not None and  errorsFittedShapeVertices is not None:
            directory = resultDir + 'shapeverticeserrors_fitted-shapeverticesserrors'
            # Show scatter correlations with occlusions.
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            scat = ax.scatter(errorsShapeVertices, errorsFittedShapeVertices, s=20, vmin=0, vmax=100,
                              c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
            for test_i in range(len(errorsLightCoeffsC)):
                if errorsFittedShapeVertices[test_i] > errorsShapeVertices[test_i] + 0.75:
                    plt.annotate(
                        str(test_i),
                        xy=(errorsShapeVertices[test_i], errorsFittedShapeVertices[test_i]),
                        xytext=(15, 15),
                        textcoords='offset points', ha='right', va='bottom',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
            cbar = fig.colorbar(scat, ticks=[0, 50, 100])
            cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
            ax.set_xlabel('Predicted shape vertices errors')
            ax.set_ylabel('Fitted shape vertices errors')
            x1, x2 = ax.get_xlim()
            y1, y2 = ax.get_ylim()
            ax.set_xlim((0, max(x2,y2)))
            ax.set_ylim((0, max(x2,y2)))
            ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
            ax.set_title('Recognition vs fitted Shape parameters errors')
            fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
            plt.close(fig)

    directory = resultDir + 'vColorsE_fitted-vColorsE'

    # Show scatter correlations with occlusions.
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(errorsVColorsE, errorsFittedVColorsE, s=20, vmin=0, vmax=100, c=testOcclusions * 100,
                      cmap=matplotlib.cm.plasma)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted VColor E coefficients errors')
    ax.set_ylabel('Fitted VColor E coefficients errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted vertex color errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'vColorsC_fitted-vColorsC'

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(errorsVColorsC, errorsFittedVColorsC, s=20, vmin=0, vmax=100, c=testOcclusions * 100,
                      cmap=matplotlib.cm.plasma)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted vertex color errors')
    ax.set_ylabel('Fitted vertex color errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted vertex color errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'vColorsS_fitted-vColorsS'

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(errorsVColorsS, errorsFittedVColorsS, s=20, vmin=0, vmax=100, c=testOcclusions * 100,
                      cmap=matplotlib.cm.plasma)
    for test_i in range(len(errorsVColorsS)):
        if errorsFittedVColorsS[test_i] > errorsVColorsS[test_i] + 0.1:
            plt.annotate(
                str(test_i),
                xy=(errorsVColorsS[test_i], errorsFittedVColorsS[test_i]),
                xytext=(15, 15),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted VColor coefficients errors')
    ax.set_ylabel('Fitted VColor coefficients errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2, y2)))
    ax.set_ylim((0, max(x2, y2)))
    ax.plot([0, max(x2, y2)], [0, max(x2, y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted vertex color errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

def saveScatterPlotsMethodFit(methodFitNum, resultDir, testOcclusions, useShapeModel, errorsPosePred, errorsPoseFitted,errorsLightCoeffsC,errorsFittedLightCoeffsC,errorsEnvMap,errorsFittedEnvMap,errorsLightCoeffs,errorsFittedLightCoeffs,errorsShapeParams,errorsFittedShapeParams,errorsShapeVertices,errorsFittedShapeVertices,errorsVColorsE,errorsFittedVColorsE,errorsVColorsC,errorsFittedVColorsC, errorsVColorsS,errorsFittedVColorsS):

    latexify(columns=2)

    directory = resultDir + 'pred-azimuth-errors_fitted-azimuth-error-' + str(methodFitNum) + '-'
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(abs(errorsPosePred[0]), abs(errorsPoseFitted[0]), s=20, vmin=0, vmax=100,
                      c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted azimuth errors')
    ax.set_ylabel('Fitted azimuth errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    for test_i in range(len(errorsPosePred[0])):
        if abs(errorsPoseFitted[0][test_i]) > abs(errorsPosePred[0][test_i]) + 10:
            plt.annotate(
                str(test_i),
                xy=(abs(errorsPosePred[0][test_i]), abs(errorsPoseFitted[0][test_i])), xytext=(0, 15),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    ax.set_title('Recognition vs fitted azimuth errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'pred-elevation-errors_fitted-elevation-error-' + str(methodFitNum) + '-'
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(abs(errorsPosePred[1]), abs(errorsPoseFitted[1]), s=20, vmin=0, vmax=100,
                      c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
    for test_i in range(len(errorsPosePred[1])):
        if abs(errorsPoseFitted[1][test_i]) > abs(errorsPosePred[1][test_i]) + 10:
            plt.annotate(
                str(test_i),
                xy=(abs(errorsPosePred[1][test_i]), abs(errorsPoseFitted[1][test_i])), xytext=(15, 15),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),

    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted elevation errors')
    ax.set_ylabel('Fitted elevation errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted azimuth errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'shcoeffsserrorsC_fitted-shcoeffsserrorsC-' + str(methodFitNum) + '-'

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(np.mean(errorsLightCoeffsC, axis=1), np.mean(errorsFittedLightCoeffsC, axis=1), s=20,
                      vmin=0, vmax=100, c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
    for test_i in range(len(errorsLightCoeffsC)):
        if np.mean(errorsFittedLightCoeffsC, axis=1)[test_i] > np.mean(errorsLightCoeffsC, axis=1)[test_i] + 0.1:
            plt.annotate(
                str(test_i),
                xy=(np.mean(errorsLightCoeffsC, axis=1)[test_i], np.mean(errorsFittedLightCoeffsC, axis=1)[test_i]), xytext=(0, 15),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted SH coefficients errors')
    ax.set_ylabel('Fitted SH coefficients errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted SH coefficients errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    if errorsEnvMap is not None and  errorsFittedEnvMap is not None:
        directory = resultDir + 'shcoeffsserrorsC_fitted-shEnvMap-' + str(methodFitNum) + '-'
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(errorsEnvMap, errorsFittedEnvMap, s=20,
                          vmin=0, vmax=100, c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted SH coefficients errors')
        ax.set_ylabel('Fitted SH coefficients errors')
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_xlim((0, max(x2,y2)))
        ax.set_ylim((0, max(x2,y2)))
        ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
        ax.set_title('Recognition vs fitted SH Environment map errors')
        fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
        plt.close(fig)

    directory = resultDir + 'shcoeffsserrors_fitted-shcoeffsserrors-' + str(methodFitNum) + '-'
    # Show scatter correlations with occlusions.
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(np.mean(errorsLightCoeffs, axis=1), np.mean(errorsFittedLightCoeffs, axis=1), s=20,
                      vmin=0, vmax=100, c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted SH coefficients errors')
    ax.set_ylabel('Fitted SH coefficients errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted SH coefficients errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    if useShapeModel:
        directory = resultDir + 'shapeparamserrors_fitted-shapeparamserrors-' + str(methodFitNum) + '-'
        # Show scatter correlations with occlusions.
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(np.mean(errorsShapeParams, axis=1), np.mean(errorsFittedShapeParams, axis=1), s=20,
                          vmin=0, vmax=100, c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted shape parameters errors')
        ax.set_ylabel('Fitted shape parameters errors')
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_xlim((0, max(x2,y2)))
        ax.set_ylim((0, max(x2,y2)))
        ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
        ax.set_title('Recognition vs fitted Shape parameters errors')
        fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
        plt.close(fig)

        if errorsShapeVertices is not None and  errorsFittedShapeVertices is not None:
            directory = resultDir + 'shapeverticeserrors_fitted-shapeverticesserrors-' + str(methodFitNum) + '-'
            # Show scatter correlations with occlusions.
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            scat = ax.scatter(errorsShapeVertices, errorsFittedShapeVertices, s=20, vmin=0, vmax=100,
                              c=testOcclusions * 100, cmap=matplotlib.cm.plasma)
            for test_i in range(len(errorsLightCoeffsC)):
                if errorsFittedShapeVertices[test_i] > errorsShapeVertices[test_i] + 0.75:
                    plt.annotate(
                        str(test_i),
                        xy=(errorsShapeVertices[test_i], errorsFittedShapeVertices[test_i]),
                        xytext=(15, 15),
                        textcoords='offset points', ha='right', va='bottom',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
            cbar = fig.colorbar(scat, ticks=[0, 50, 100])
            cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
            ax.set_xlabel('Predicted shape vertices errors')
            ax.set_ylabel('Fitted shape vertices errors')
            x1, x2 = ax.get_xlim()
            y1, y2 = ax.get_ylim()
            ax.set_xlim((0, max(x2,y2)))
            ax.set_ylim((0, max(x2,y2)))
            ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
            ax.set_title('Recognition vs fitted Shape parameters errors')
            fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
            plt.close(fig)

    directory = resultDir + 'vColorsE_fitted-vColorsE-' + str(methodFitNum) + '-'

    # Show scatter correlations with occlusions.
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(errorsVColorsE, errorsFittedVColorsE, s=20, vmin=0, vmax=100, c=testOcclusions * 100,
                      cmap=matplotlib.cm.plasma)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted VColor E coefficients errors')
    ax.set_ylabel('Fitted VColor E coefficients errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted vertex color errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'vColorsC_fitted-vColorsC-' + str(methodFitNum) + '-'

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(errorsVColorsC, errorsFittedVColorsC, s=20, vmin=0, vmax=100, c=testOcclusions * 100,
                      cmap=matplotlib.cm.plasma)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted vertex color errors')
    ax.set_ylabel('Fitted vertex color errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2,y2)))
    ax.set_ylim((0, max(x2,y2)))
    ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted vertex color errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'vColorsS_fitted-vColorsS-' + str(methodFitNum) + '-'

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(errorsVColorsS, errorsFittedVColorsS, s=20, vmin=0, vmax=100, c=testOcclusions * 100,
                      cmap=matplotlib.cm.plasma)
    for test_i in range(len(errorsVColorsS)):
        if errorsFittedVColorsS[test_i] > errorsVColorsS[test_i] + 0.1:
            plt.annotate(
                str(test_i),
                xy=(errorsVColorsS[test_i], errorsFittedVColorsS[test_i]),
                xytext=(15, 15),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), size=8)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Predicted VColor coefficients errors')
    ax.set_ylabel('Fitted VColor coefficients errors')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, max(x2, y2)))
    ax.set_ylim((0, max(x2, y2)))
    ax.plot([0, max(x2, y2)], [0, max(x2, y2)], ls="--", c=".3")
    ax.set_title('Recognition vs fitted vertex color errors')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

def saveLikelihoodPlots(resultDir, occlusions, methodsPred, plotColors, plotMethodsIndices, meanLikelihoodArr):

    latexify(columns=2)
    directory = resultDir + 'predictionLikelihood-FitLikelihood'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(occlusions, meanLikelihoodArr[0], c='b', label='Ground-truth NLL')
    ax.plot(occlusions, meanLikelihoodArr[1], c='r', label='Robust Fit NLL')
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Mean Likelihood')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    # ax.set_xlim((min(x1), 100))
    # ax.set_ylim((-0.0, y2))
    ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'predictionLikelihoodGaussian-FitLikelihoodGaussian'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(occlusions, meanLikelihoodArr[2], c='b', label='Groundtruth NLL')
    ax.plot(occlusions, meanLikelihoodArr[3], c='r', label='Robust Fit NLL')
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Mean likelihood')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    # ax.set_xlim((min(x1), 100))
    # ax.set_ylim((-0.0, y2))
    ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-plot.pdf', bbox_inches='tight')
    plt.close(fig)


def saveLikelihoodScatter(resultDirOcclusion, testSet, testOcclusions,  likelihoods):
    # Show scatter correlations with occlusions.

    directory = resultDirOcclusion + 'robustLikelihood_fitted-robustLikelihood'
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    scat = ax.scatter(likelihoods[0][testSet], likelihoods[1][testSet], s=20, vmin=0, vmax=100, c=testOcclusions * 100,
                      cmap=matplotlib.cm.plasma)
    cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    ax.set_xlabel('Ground-truth NLL (avg)')
    ax.set_ylabel('Recognition+Fit NLL (avg)')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((min(x1,y1), max(x2,y2)))
    ax.set_ylim((min(x1,y1), max(x2,y2)))
    ax.plot([min(x1,y1), max(x2,y2)], [min(x1,y1), max(x2,y2)], ls="--", c=".3")
    ax.set_title('GT vs Recognition+Robust fit negative log-likelihood')
    fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    # directory = resultDirOcclusion + 'gaussianLikelihood_fitted-gaussianLikelihood'
    # fig = plt.figure()
    # ax = fig.add_subplot(111, aspect='equal')
    # scat = ax.scatter(likelihoods[2][testSet], likelihoods[3][testSet], s=20, vmin=0, vmax=100, c=testOcclusions * 100,
    #                   cmap=matplotlib.cm.plasma)
    # cbar = fig.colorbar(scat, ticks=[0, 50, 100])
    # cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
    # ax.set_xlabel('Ground-truth avg likelihood')
    # ax.set_ylabel('Fitted avg likelihood')
    # x1, x2 = ax.get_xlim()
    # y1, y2 = ax.get_ylim()
    # ax.set_xlim((0, max(x2,y2)))
    # ax.set_ylim((0, max(x2,y2)))
    # ax.plot([0, max(x2,y2)], [0, max(x2,y2)], ls="--", c=".3")
    # ax.set_title('GT vs Recognition+Gaussian Fit likelihood')
    # fig.savefig(directory + '-performance-scatter.pdf', bbox_inches='tight')
    # plt.close(fig)

def saveHistograms():
    pass
    # latexify(columns=2)

    # directory = resultDir + 'reconition-azimuth'
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hist(errorsPosePredList[2][0], bins=40)
    # ax.set_xlabel('Azimuth errors')
    # ax.set_ylabel('Counts')
    # ax.set_title('Azimuth errors histogram')
    # ax.set_xlim((-180, 180))
    # fig.savefig(directory + '-histogram.pdf', bbox_inches='tight')
    # plt.close(fig)
    #
    # directory = resultDir + 'robust-azimuth'
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hist(errorsPosePredList[4][0], bins=40)
    # ax.set_xlabel('Azimuth errors')
    # ax.set_ylabel('Counts')
    # ax.set_xlim((-180, 180))
    # ax.set_title('Azimuth errors histogram')
    # fig.savefig(directory + '-histogram.pdf', bbox_inches='tight')
    # plt.close(fig)
    #
    # directory = resultDir + 'robust-elevation'
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hist(errorsPosePredList[4][1], bins=100)
    # ax.set_xlabel('Elevation errors')
    # ax.set_ylabel('Counts')
    # x1, x2 = ax.get_xlim()
    # y1, y2 = ax.get_ylim()
    # ax.set_xlim((-90, 90))
    # ax.set_title('Elevation errors histogram')
    # fig.savefig(directory + '-histogram.pdf', bbox_inches='tight')
    # plt.close(fig)
    #
    #
    # directory = resultDir + 'recognition-elevation'
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hist(errorsPosePredList[2][1], bins=30)
    # ax.set_xlabel('Elevation errors')
    # ax.set_ylabel('Counts')
    # ax.set_title('Elevation errors histogram')
    # ax.set_xlim((-90, 90))
    # ax.set_ylim((y1, y2))
    # fig.savefig(directory + '-histogram.pdf', bbox_inches='tight')
    # plt.close(fig)

    # directory = resultDir + 'robust-elevation-occlusion'
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # scat = ax.scatter(testOcclusions*100, abs(errorsPosePredList[4][1]), s=20, vmin=0, vmax=100)
    # ax.set_xlabel('Occlusion level (\%)')
    # ax.set_ylabel('Robust fit elevation errors')
    # x1, x2 = ax.get_xlim()
    # y1, y2 = ax.get_ylim()
    # ax.set_xlim((0, max(x2,y2)))
    # ax.set_ylim((0, max(x2,y2)))
    # ax.set_title('Fitted azimuth errors')
    # fig.savefig(directory + '-scatter.pdf', bbox_inches='tight')
    # plt.close(fig)
    #
    # directory = resultDir + 'recognition-elevation-occlusion'
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # scat = ax.scatter(testOcclusions*100, abs(errorsPosePredList[2][1]), s=20, vmin=0, vmax=100)
    # ax.set_ylabel('Robust fit elevation errors')
    # ax.set_xlabel('Occlusion level (\%)')
    # x1, x2 = ax.get_xlim()
    # y1, y2 = ax.get_ylim()
    # ax.set_xlim((0, max(x2,y2)))
    # ax.set_ylim((0, max(x2,y2)))
    # ax.set_title('Fitted azimuth errors')
    # fig.savefig(directory + '-scatter.pdf', bbox_inches='tight')
    # plt.close(fig)

def saveConditionalHistograms(resultDir, occlusions, methodsPred, variablesDescr, plotMethodsIndices, errorsList):
    latexify(columns=2)

    numBins = 20
    numErrBins = 40


    for variable_i in range(len(variablesDescr)):
        variableDescr = variablesDescr[variable_i]
        errors = errorsList[variable_i]

        directory = resultDir + variableDescr

        maxErr = np.max(np.array([np.max(errors[method_i]) if errors[method_i] is not None else 0 for method_i in plotMethodsIndices]))


        for method_i in plotMethodsIndices:
            condHists = []
            if errors[method_i] is not None and len(errors[method_i]) > 0:
                fig = plt.figure()
                ax = fig.add_subplot(111)

                bins = np.linspace(0,0.9,numBins)

                occlusionBinIndices = np.array([(occlusions < bins[i]) & (occlusions > bins[i-1]) for i in np.arange(1,len(bins))])

                # ax.plot(occlusions, errorsPosePredList[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
                legend = ax.legend(loc='best')
                ax.set_xlabel('Occlusion (\%)')
                ax.set_ylabel('Error Cond. Histogram')
                # x1, x2 = ax.get_xlim()
                # y1, y2 = ax.get_ylim()



                if variableDescr == 'Illumination':
                    maxErr = 0.2

                for i in range(len(bins)-1):

                    hist, bin_edges = np.histogram(errors[method_i][occlusionBinIndices[i]], numErrBins, range=(0,maxErr), density=True)
                    condHists = condHists + [hist]

                # ipdb.set_trace()
                histMat = np.flipud(np.vstack(condHists).T)

                maxHistProb = np.max(histMat)

                ax.matshow(histMat, cmap=matplotlib.cm.gray, interpolation='none', extent=[0,90,0,45], vmin=0, vmax=maxHistProb)

                # locs, labels = ax.xticks()

                # ax.set_xticks(bins)

                ax.set_xlim((0, 90))
                ax.set_ylim((0, 45))

                yticks = np.linspace(0,45,10)

                ytickslabels = ["{:1.2f}".format(num) for num in np.linspace(0, maxErr, 10)]
                ax.set_yticks(yticks)
                ax.set_yticklabels(ytickslabels)
                # ax.set_xlim((0, 100))

                # ax.set_ylim((-0.0, y2))

                # ax.set_title('Cumulative segmentation accuracy per occlusion level')
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.tick_params(axis='both', which='minor', labelsize=8)

                fig.savefig(directory + '-' + methodsPred[method_i] + '-conditional-histogram.pdf', bbox_inches='tight')
                plt.close(fig)




def saveOcclusionPlots(resultDir, prefix, occlusions, methodsPred, plotColors, plotStyles, plotMethodsIndices, useShapeModel, meanAbsErrAzsArr, meanAbsErrElevsArr, meanErrorsVColorsCArr, meanErrorsVColorsEArr, meanErrorsVColorsSArr, meanErrorsLightCoeffsArr, meanErrorsShapeParamsArr, meanErrorsShapeVerticesArr, meanErrorsLightCoeffsCArr, meanErrorsEnvMapArr,meanErrorsSegmentationArr):

    latexify(columns=2)

    directory = resultDir + prefix + 'predictionMeanError-Segmentation'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        if meanErrorsSegmentationArr[method_i] is not None and len(meanErrorsSegmentationArr[method_i]) > 0:
            ax.plot(occlusions, meanErrorsSegmentationArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Segmentation accuracy')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    # ax.set_title('Cumulative segmentation accuracy per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)


    directory = resultDir + prefix + 'predictionMeanError-Azimuth'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        if meanAbsErrAzsArr[method_i] is not None and len(meanAbsErrAzsArr[method_i]) > 0:
            ax.semilogy(occlusions, meanAbsErrAzsArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Angular error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((y1, y2))
    # ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + prefix +'predictionMeanError-Elev'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        if meanAbsErrElevsArr[method_i] is not None and len(meanAbsErrElevsArr[method_i]) > 0:
            ax.plot(occlusions, meanAbsErrElevsArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Angular error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    # ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + prefix +'predictionMeanError-VColors-C'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        if meanErrorsVColorsCArr[method_i] is not None and len(meanErrorsVColorsCArr[method_i]) > 0:
            ax.plot(occlusions, meanErrorsVColorsCArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Appearance error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    # ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + prefix +'predictionMeanError-VColors-E'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        if meanErrorsVColorsEArr[method_i] is not None and len(meanErrorsVColorsEArr[method_i]) > 0:
            ax.plot(occlusions, meanErrorsVColorsEArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Appearance error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    # ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)


    directory = resultDir + prefix +'predictionMeanError-VColors-S'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        if meanErrorsVColorsSArr[method_i] is not None and len(meanErrorsVColorsSArr[method_i]) > 0:
            ax.plot(occlusions, meanErrorsVColorsSArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Appearance error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    # ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)


    directory = resultDir + prefix +'predictionMeanError-SH'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        if meanErrorsLightCoeffsArr[method_i] is not None and len(meanErrorsLightCoeffsArr[method_i]) > 0:
            ax.plot(occlusions, meanErrorsLightCoeffsArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Illumination error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    # ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)

    if useShapeModel:
        directory = resultDir + prefix +'predictionMeanError-ShapeParams'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for method_i in plotMethodsIndices:
            if meanErrorsShapeParamsArr[method_i] is not None and len(meanErrorsShapeParamsArr[method_i]) > 0:
                ax.plot(occlusions, meanErrorsShapeParamsArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
        legend = ax.legend(loc='best')
        ax.set_xlabel('Occlusion (\%)')
        ax.set_ylabel('Shape parameters error')

        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_xlim((0, 100))
        ax.set_ylim((-0.0, y2))
        # ax.set_title('Cumulative prediction per occlusion level')
        fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
        plt.close(fig)

        directory = resultDir + prefix +'predictionMeanError-ShapeVertices'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for method_i in plotMethodsIndices:
            if meanErrorsShapeVerticesArr[method_i] is not None and len(meanErrorsShapeVerticesArr[method_i]) > 0:
                ax.semilogy(occlusions, meanErrorsShapeVerticesArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
        legend = ax.legend(loc='best')
        ax.set_xlabel('Occlusion (\%)')
        ax.set_ylabel('Shape vertices error')

        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_xlim((0, 100))
        ax.set_ylim((y1, y2))
        # ax.set_title('Cumulative prediction per occlusion level')
        fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
        plt.close(fig)

    directory = resultDir + prefix +'predictionMeanError-SH-C'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        if meanErrorsLightCoeffsCArr[method_i] is not None and len(meanErrorsLightCoeffsCArr[method_i]) > 0:
            ax.plot(occlusions, meanErrorsLightCoeffsCArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Illumination error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    # ax.set_title('Cumulative prediction per occlusion level')
    fig.savefig(directory + '-performance-plot.pdf', bbox_inches='tight')
    plt.close(fig)


    directory = resultDir + prefix +'predictionMeanError-SH-EnvMap'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_i in plotMethodsIndices:
        if meanErrorsEnvMapArr[method_i] is not None and len(meanErrorsEnvMapArr[method_i]) > 0:
            ax.plot(occlusions, meanErrorsEnvMapArr[method_i], c=plotColors[method_i], linestyle=plotStyles[method_i], label=methodsPred[method_i])
    legend = ax.legend(loc='best')
    ax.set_xlabel('Occlusion (\%)')
    ax.set_ylabel('Illumination error')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.set_xlim((0, 100))
    ax.set_ylim((-0.0, y2))
    # ax.set_title('Cumulative prediction per occlusion level')
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

def computeErrors(setTest, azimuths, testAzsRel, elevations, testElevsGT, vColors, testVColorGT, lightCoeffs, testLightCoefficientsGTRel, approxProjections,  approxProjectionsGT, shapeParams, testShapeParamsGT, useShapeModel, chShapeParams, chVertices, posteriors, masksGT):
    errorsPosePredList = []
    errorsLightCoeffsList = []
    errorsShapeParamsList = []
    errorsShapeVerticesList = []
    errorsEnvMapList = []
    errorsLightCoeffsCList = []
    errorsVColorsEList = []
    errorsVColorsCList = []
    errorsVColorsSList = []
    errorsSegmentation = []

    for method in range(len(azimuths)):

        print("Computing errors for method " + str(method))
        azsPred = azimuths[method]
        elevsPred = elevations[method]
        vColorsPred = vColors[method]
        posteriorsPred = posteriors[method]
        relLightCoefficientsPred = lightCoeffs[method]
        if approxProjectionsGT is not None:
            approxProjectionsPred = approxProjections[method]

        if useShapeModel:
            shapeParamsPred = shapeParams[method]

        if azsPred is not None:
            errorsPosePred = recognition_models.evaluatePrediction(testAzsRel[setTest], testElevsGT[setTest], azsPred[setTest], elevsPred[setTest])
            errorsPosePredList = errorsPosePredList + [errorsPosePred]
        else:
            errorsPosePredList = errorsPosePredList + [None]

        if relLightCoefficientsPred is not None:
            errorsLightCoeffs = (testLightCoefficientsGTRel[setTest] - relLightCoefficientsPred[setTest]) ** 2
            errorsLightCoeffsList = errorsLightCoeffsList + [errorsLightCoeffs]
        else:
            errorsLightCoeffsList = errorsLightCoeffsList + [None]

        if useShapeModel and shapeParamsPred is not None:
            errorsShapeParams = (testShapeParamsGT[setTest] - shapeParamsPred[setTest]) ** 2
            errorsShapeParamsList = errorsShapeParamsList + [errorsShapeParams]

            errorsShapeVertices = shapeVertexErrors(chShapeParams, chVertices, testShapeParamsGT[setTest], shapeParamsPred[setTest])
            errorsShapeVerticesList = errorsShapeVerticesList + [errorsShapeVertices]
        else:
            errorsShapeParamsList = errorsShapeParamsList + [None]
            errorsShapeVerticesList = errorsShapeVerticesList + [None]


        if approxProjectionsGT is not None and approxProjectionsPred is not None:

            envMapProjScaling = scaleInvariantMSECoeff(approxProjectionsPred.reshape([len(approxProjectionsPred), -1])[setTest], approxProjectionsGT.reshape([len(approxProjectionsGT), -1])[setTest])
            errorsEnvMap = np.mean((approxProjectionsGT[setTest] -  envMapProjScaling[:,None, None]*approxProjectionsPred[setTest])**2, axis=(1,2))

            errorsEnvMapList=  errorsEnvMapList + [errorsEnvMap]
        else:
            errorsEnvMapList = errorsEnvMapList + [None]

        if relLightCoefficientsPred is not None:

            envMapScaling = scaleInvariantMSECoeff(relLightCoefficientsPred[setTest], testLightCoefficientsGTRel[setTest])
            errorsLightCoeffsC = (testLightCoefficientsGTRel[setTest] - envMapScaling[:,None]* relLightCoefficientsPred[setTest]) ** 2

            errorsLightCoeffsCList = errorsLightCoeffsCList + [errorsLightCoeffsC]
        else:
            errorsLightCoeffsCList = errorsLightCoeffsCList + [None]

        if vColorsPred is not None:
            errorsVColorsE = image_processing.eColourDifference(testVColorGT[setTest], vColorsPred[setTest])
            errorsVColorsEList = errorsVColorsEList + [errorsVColorsE]

            errorsVColorsC = image_processing.cColourDifference(testVColorGT[setTest], vColorsPred[setTest])
            errorsVColorsCList = errorsVColorsCList + [errorsVColorsC]

            errorsVColorsS = image_processing.scaleInvariantColourDifference(testVColorGT[setTest], vColorsPred[setTest])
            errorsVColorsSList = errorsVColorsSList + [errorsVColorsS]
        else:
            errorsVColorsEList = errorsVColorsEList + [None]
            errorsVColorsCList = errorsVColorsCList + [None]
            errorsVColorsSList = errorsVColorsSList + [None]

        if posteriorsPred is not None:
            # if method == 3:
            #     ipdb.set_trace()
            masksCat = np.concatenate([masksGT[:,:,:,None][setTest], posteriorsPred[:,:,:,None][setTest]], axis=3)
            errorSegmentation = np.sum(np.all(masksCat, axis=3), axis=(1,2)) / np.sum(np.any(masksCat, axis=3), axis=(1,2))
            errorsSegmentation = errorsSegmentation + [errorSegmentation]

        else:
            errorsSegmentation = errorsSegmentation + [None]

    return errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsVColorsSList, errorsSegmentation

def computeErrorAverages(averageFun, testSet, useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsVColorsSList, errorsSegmentation):
    meanAbsErrAzsList = []
    meanAbsErrElevsList = []
    meanErrorsLightCoeffsList = []
    meanErrorsShapeParamsList = []
    meanErrorsShapeVerticesList = []
    meanErrorsLightCoeffsCList = []
    meanErrorsEnvMapList = []
    meanErrorsVColorsEList = []
    meanErrorsVColorsCList = []
    meanErrorsVColorsSList = []
    meanErrorsSegmentation = []

    for method_i in range(len(errorsPosePredList)):
        if errorsPosePredList[method_i] is not None:
            meanAbsErrAzsList = meanAbsErrAzsList + [averageFun(np.abs(errorsPosePredList[method_i][0][testSet]))]
        else:
            meanAbsErrAzsList = meanAbsErrAzsList + [None]

        if errorsPosePredList[method_i] is not None:
            meanAbsErrElevsList = meanAbsErrElevsList + [averageFun(np.abs(errorsPosePredList[method_i][1][testSet]))]
        else:
            meanAbsErrElevsList  = meanAbsErrElevsList  + [None]

        if errorsLightCoeffsList[method_i] is not None:
            meanErrorsLightCoeffsList = meanErrorsLightCoeffsList + [averageFun(averageFun(errorsLightCoeffsList[method_i][testSet],axis=1), axis=0)]
        else:
            meanErrorsLightCoeffsList = meanErrorsLightCoeffsList + [None]

        if useShapeModel:
            if errorsShapeParamsList[method_i] is not None:
                meanErrorsShapeParamsList = meanErrorsShapeParamsList + [averageFun(np.mean(errorsShapeParamsList[method_i][testSet],axis=1), axis=0)]
            else:
                meanErrorsShapeParamsList = meanErrorsShapeParamsList + [None]

            if errorsShapeVerticesList[method_i] is not None:
                meanErrorsShapeVerticesList = meanErrorsShapeVerticesList + [averageFun(errorsShapeVerticesList[method_i][testSet], axis=0)]
            else:
                meanErrorsShapeVerticesList = meanErrorsShapeVerticesList + [None]

        if errorsLightCoeffsCList[method_i] is not None:
            meanErrorsLightCoeffsCList = meanErrorsLightCoeffsCList + [averageFun(np.mean(errorsLightCoeffsCList[method_i][testSet],axis=1), axis=0)]
        else:
            meanErrorsLightCoeffsCList = meanErrorsLightCoeffsCList + [None]

        if errorsEnvMapList[method_i] is not None:
            meanErrorsEnvMapList = meanErrorsEnvMapList + [averageFun(errorsEnvMapList[method_i][testSet])]
        else:
            meanErrorsEnvMapList = meanErrorsEnvMapList + [None]

        if errorsVColorsEList[method_i] is not None:
            meanErrorsVColorsEList = meanErrorsVColorsEList + [averageFun(errorsVColorsEList[method_i][testSet], axis=0)]
        else:
            meanErrorsVColorsEList = meanErrorsVColorsEList + [None]

        if errorsVColorsCList[method_i] is not None:
            meanErrorsVColorsCList = meanErrorsVColorsCList + [averageFun(errorsVColorsCList[method_i][testSet], axis=0)]
        else:
            meanErrorsVColorsCList = meanErrorsVColorsCList + [None]

        if errorsVColorsSList[method_i] is not None:
            meanErrorsVColorsSList = meanErrorsVColorsSList + [averageFun(errorsVColorsSList[method_i][testSet], axis=0)]
        else:
            meanErrorsVColorsSList = meanErrorsVColorsSList + [None]


        if errorsSegmentation[method_i] is not None:
            meanErrorsSegmentation = meanErrorsSegmentation + [averageFun(errorsSegmentation[method_i][testSet], axis=0)]
        else:
            meanErrorsSegmentation = meanErrorsSegmentation + [None]


    return meanAbsErrAzsList, meanAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList, meanErrorsVColorsSList, meanErrorsSegmentation


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

def loadMasksMug(imagesDir, maskSet):
    masks = []
    for imageit, imageid  in enumerate(maskSet):

        if os.path.isfile(imagesDir + 'mask' + str(imageid) + '_mug.npy'):
            masks = masks + [np.load(imagesDir + 'mask' + str(imageid) + '_mug.npy')[None,:,:]]

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


def getTriplets(parameterVals, closeDist = 5*np.pi/180, farDist=15*np.pi/180, normConst = 2*np.pi, chunkSize=10):
    numTrainSet = len(parameterVals)

    initIdx = np.random.permutation(np.arange(numTrainSet))
    orderedIdx = np.argsort(parameterVals)

    chunkArange = np.arange(chunkSize)
    shuffledIdx = np.arange(numTrainSet)
    for i in np.arange(0, numTrainSet, chunkSize):
        maxi = min(numTrainSet, i + chunkSize)

        shuffledIdx[i:i+chunkSize] = np.random.permutation(chunkArange)[:maxi] + i

    idxDist5 = np.roll(np.arange(numTrainSet), int(numTrainSet * closeDist / normConst) )
    idxDist15 = np.roll(np.arange(numTrainSet), int(numTrainSet * farDist / normConst) )

    anchorIdx = orderedIdx[initIdx]
    closeIdx = orderedIdx[shuffledIdx[idxDist5][initIdx]]
    farIdx = orderedIdx[shuffledIdx[idxDist15][initIdx]]

    return anchorIdx, closeIdx, farIdx


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
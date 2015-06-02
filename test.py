#!/usr/bin/env python3.4m
import matplotlib
# matplotlib.use('Agg')
import sceneimport
from utils import *
from score_image import *
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import cv2

numpy.random.seed(1)

inchToMeter = 0.0254

cam = bpy.data.cameras.new("MainCamera")
camera = bpy.data.objects.new("MainCamera", cam)

world = bpy.data.worlds.new("MainWorld")

rendersDir = 'output/'

groundTruth, imageFiles, segmentFiles, prefixes = loadGroundTruth(rendersDir)

width = 110
height = 110

useCycles = False
useGPU = False
distance = 0.45
numSamples = 16

originalLoc = mathutils.Vector((0,-distance , 0))

numpy.random.seed(1)

minThresTemplate = 10
maxThresTemplate = 100
minThresImage = 50
maxThresImage = 150

baseDir = '../databaseFull/models/'

experimentTeapots = [2]

outputExperiments = []

# distanceTypes = ['chamferDataToModel', 'robustChamferDataToModel', 'sqDistImages', 'robustSqDistImages']
distanceTypes = ['sqDistImages', 'negLogLikelihood','negLogLikelihoodRobust']
masks = numpy.array([])
segmentImages = []
for segment in segmentFiles:
    print(segment)
    segmentImg = cv2.imread(segment, cv2.IMREAD_ANYDEPTH)/255.0
    segmentImg = segmentImg[..., numpy.newaxis]
    segmentImages = segmentImages + [segmentImg]

masks = numpy.concatenate([aux for aux in segmentImages], axis=-1)

layerPrior = globalLayerPrior(masks)

backgroundModels = [False, True]

completeScene = True
if True not in backgroundModels:
    completeScene = False

sortedGroundTruth = groundTruth[numpy.argsort(groundTruth[:,6]), :]

indicesOccluded = (sortedGroundTruth[:,5] < 0.99) & (sortedGroundTruth[:,5] > 0.05)

[targetScenes, targetModels] = sceneimport.loadTargetModels(experimentTeapots)

for teapotTest in experimentTeapots:

    # robust = True
    # robustScale = 0
    teapot = targetModels[teapotTest]
    teapot.layers[0] = True
    teapot.layers[1] = True

    print("Experiment on teapot " + teapot.name)

    indicesTeapot = (sortedGroundTruth[:, 3] == teapotTest)

    indices = numpy.where(indicesTeapot & indicesOccluded)

    selTest = indices[0]

    numTests = len(selTest)
    print ("Total of " + str(numTests) + " tests")

    # expSelTest = [1,2]
    expSelTest = numpy.arange(0,numTests,int(numTests)/500)

    # numpy.arange(0,numTests,int(numTests))
    print ("Running only " + str(len(expSelTest)) + " tests")
    # selTest = numpy.random.permutation(selTest)
    currentScene = -1

    performance = {}
    elevations = {}
    groundTruthRelAzimuths = {}
    groundTruthAzimuths = {}
    bestRelAzimuths= {}
    bestAzimuths= {}
    occlusions = {}

    for backgroundModelOut in backgroundModels:
        modelStr = 'background/'
        if not backgroundModelOut:
            modelStr = 'no_background/'
        for distanceTypeOut in distanceTypes:
            performance[(backgroundModelOut, distanceTypeOut)] = numpy.array([])
            elevations[(backgroundModelOut, distanceTypeOut)] = numpy.array([])
            groundTruthRelAzimuths[(backgroundModelOut, distanceTypeOut)] = numpy.array([])
            groundTruthAzimuths[(backgroundModelOut, distanceTypeOut)] = numpy.array([])
            bestRelAzimuths[(backgroundModelOut, distanceTypeOut)]= numpy.array([])
            bestAzimuths[(backgroundModelOut, distanceTypeOut)] = numpy.array([])
            occlusions[(backgroundModelOut, distanceTypeOut)] = numpy.array([])


    for selTestNum in expSelTest:

        test = selTest[selTestNum]

        groundTruthAz = sortedGroundTruth[test,0]
        groundTruthObjAz = sortedGroundTruth[test,1]
        groundTruthRelAz = numpy.arctan2(numpy.sin((groundTruthAz-groundTruthObjAz)*numpy.pi/180), numpy.cos((groundTruthAz-groundTruthObjAz)*numpy.pi/180))*180/numpy.pi
        groundTruthEl = sortedGroundTruth[test,2]
        occlusion = sortedGroundTruth[test,5]
        sceneNum = int(sortedGroundTruth[test,6])
        targetIndex= int(sortedGroundTruth[test,7])

        if currentScene != sceneNum:
            if currentScene != -1:
                # Cleanup
                for objnum, obji in enumerate(scene.objects):
                    if obji.name != teapot.name:
                        obji.user_clear()
                        bpy.data.objects.remove(obji)

                scene.user_clear()
                bpy.data.scenes.remove(scene)

            currentScene = sceneNum

            sceneFileName = '{0:05d}'.format(sceneNum)

            print("Experiment on scene " + sceneFileName)

            instances = sceneimport.loadScene('../databaseFull/scenes/scene' + sceneFileName + '.txt')

            targetParentPosition = instances[targetIndex][2]
            targetParentIndex = instances[targetIndex][1]

            [blenderScenes, modelInstances] = sceneimport.importBlenderScenes(instances, completeScene, targetIndex)
            # targetParentInstance = modelInstances[targetParentIndex]
            roomName = ''
            for model in modelInstances:
                reg = re.compile('(room[0-9]+)')
                res = reg.match(model.name)
                if res:
                    roomName = res.groups()[0]

            scene = sceneimport.composeScene(modelInstances, targetIndex)

            scene.update()
            scene.render.threads = 8
            scene.render.threads_mode = 'AUTO'
            bpy.context.screen.scene = scene

            cycles = bpy.context.scene.cycles
            scene.render.tile_x = 55
            scene.render.tile_y = 55

            setupScene(scene, targetIndex,roomName, world, distance, camera, width, height, numSamples, useCycles, useGPU)

            scene.objects.link(teapot)
            teapot.layers[1] = True
            scene.update()

            teapot.matrix_world = mathutils.Matrix.Translation(targetParentPosition)
            center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
            original_matrix_world = teapot.matrix_world.copy()
            teapot.matrix_world = original_matrix_world

        azimuthRot = mathutils.Matrix.Rotation(radians(-groundTruthObjAz), 4, 'Z')
        teapot.matrix_world = mathutils.Matrix.Translation(original_matrix_world.to_translation()) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world

        for backgroundModel in backgroundModels:

            variances = numpy.array([])

            modelStr = 'background/'
            if not backgroundModel:
                modelStr = 'no_background/'
            print("Experiment on background model " + modelStr)

            scene.layers[1] = True
            scene.layers[0] = False
            scene.render.layers[0].use = False
            scene.render.layers[1].use = True
            teapot.layers[1] = True

            if backgroundModel:
                scene.layers[1] = False
                scene.layers[0] = True
                scene.render.layers[0].use = True
                scene.render.layers[1].use = False
                teapot.layers[0] = True

            scene.update()

            sqDistsSeq = []
            for distanceType in distanceTypes:
                print("Experiment on model " + distanceType)
                computingSqDists = False

                if distanceType == 'sqDistImages':
                    computingSqDists = True

                if not computingSqDists:
                    sqRes = numpy.concatenate([aux[..., numpy.newaxis] for aux in sqDistsSeq], axis=-1)
                    variances = computeVariances(sqRes)
                    variances[numpy.where(variances <= 1)] = 1/255.0

                # robust = not robust
                # if robust is False:
                #     robustScale = 0

                scores = []
                relAzimuths = []
                azimuths = []

                directory = 'aztest/'  + 'teapot' + str(teapotTest)  + '/' + modelStr + distanceType
                if not os.path.exists('aztest/'  + 'teapot' + str(teapotTest)  + '/'):
                    os.makedirs('aztest/'  + 'teapot' + str(teapotTest)  + '/')

                if not os.path.exists(directory + 'test_samples'):
                    os.makedirs(directory + 'test_samples')

                numDir = directory +  'test_samples/num' + str(test) + '_azim' + str(int(groundTruthAz)) + '_elev' + str(int(groundTruthEl)) + '/'
                if not os.path.exists(numDir):
                    os.makedirs(numDir)

                testImage = cv2.imread(imageFiles[test])/255.0
                # testImage = cv2.cvtColor(numpy.float32(rgbTestImage*255), cv2.COLOR_RGB2BGR)/255.0

                testImageEdges = cv2.Canny(numpy.uint8(testImage*255), minThresImage,maxThresImage)
                cv2.imwrite(numDir + "image_canny" + ".png" , testImageEdges)
                cv2.imwrite(numDir + "image" + ".png" , numpy.uint8(testImage*255))

                score = numpy.finfo(numpy.float64).max

                elevation = groundTruthEl

                azimuth = 0
                elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')

                for azimuth in numpy.arange(0,360,5):

                    azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
                    elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
                    location = center + azimuthRot * elevationRot * originalLoc
                    camera.location = location

                    scene.update()
                    look_at(camera, center)
                    scene.update()

                    scene.render.filepath = numDir  + '_blender.png'

                    bpy.ops.render.render( write_still=True )

                    image = cv2.imread(scene.render.filepath)

                    # blendImage = bpy.data.images['Render Result']
                    # #
                    # image2 = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height,width,4]))[:,:,0:3]

                    # # ipdb.set_trace()
                    # Truncate intensities larger than 1.
                    # image = image * 1.5
                    # image2[numpy.where(image2 > 1)] = 1
                    # ipdb.set_trace()
                    # image[0:20, 75:100, :] = 0

                    # image2 = cv2.cvtColor(numpy.float32(image2), cv2.COLOR_RGB2BGR)
                    image = image/255.0
                    # image2 = image2/255.0

                    methodParams = {'variances':variances, 'layerPrior': layerPrior, 'minThresImage': minThresImage, 'maxThresImage': maxThresImage, 'minThresTemplate': minThresTemplate, 'maxThresTemplate': maxThresTemplate}

                    distance = scoreImage(testImage, image, distanceType, methodParams)
                    # cv2.imwrite(numDir + 'image' + "_az" + '%.1f' % azimuth + '_dist' + '%.1f' % distance + '.png', numpy.uint8(image*255.0))

                    scores.append(distance)
                    relAzimuth = numpy.mod(numpy.arctan2(numpy.sin((azimuth-groundTruthObjAz)*numpy.pi/180), numpy.cos((azimuth-groundTruthObjAz)*numpy.pi/180))*180/numpy.pi , 360)
                    azimuths.append(azimuth)

                    if distance <= score:
                        imageEdges = cv2.Canny(numpy.uint8(image*255.0), minThresTemplate,maxThresTemplate)
                        bestImageEdges = imageEdges
                        bestImage = image
                        score = distance
                        bestRelAzimuth = relAzimuth
                        bestAzimuth = azimuth

                # if robust is False:
                #     robustScale = 1.4826 * numpy.sqrt(numpy.median(scores))

                # error = numpy.arctan2(numpy.sin((groundTruthRelAz-bestRelAzimuth)*numpy.pi/180), numpy.cos((groundTruthRelAz-bestRelAzimuth)*numpy.pi/180))*180/numpy.pi
                error = numpy.arctan2(numpy.sin((groundTruthAz-bestAzimuth)*numpy.pi/180), numpy.cos((groundTruthAz-bestAzimuth)*numpy.pi/180))*180/numpy.pi
                performance[(backgroundModel, distanceType)] = numpy.append(performance[(backgroundModel, distanceType)], error)
                elevations[(backgroundModel, distanceType)] = numpy.append(elevations[(backgroundModel, distanceType)], elevation)
                bestAzimuths[(backgroundModel, distanceType)] = numpy.append(bestAzimuths[(backgroundModel, distanceType)], bestAzimuth)
                groundTruthAzimuths[(backgroundModel, distanceType)] = numpy.append(groundTruthAzimuths[(backgroundModel, distanceType)], groundTruthAz)
                occlusions[(backgroundModel, distanceType)] = numpy.append(occlusions[(backgroundModel, distanceType)], occlusion)

                sqDist = sqDistImages(bestImage, testImage)
                disp = cv2.normalize(sqDist, sqDist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(numDir + 'sqDists' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))

                if computingSqDists:
                    sqDistsSeq = sqDistsSeq + [sqDist]

                if distanceType == 'negLogLikelihoodRobust':
                    pixLik = pixelLikelihoodRobust(testImage, bestImage, layerPrior, variances)
                    plt.imshow(pixLik)
                    plt.colorbar()
                    plt.savefig(numDir + 'pixelLikelihoodRobustPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    plt.clf()

                    disp = cv2.normalize(pixLik, pixLik, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imwrite(numDir + 'pixelLikelihoodRobust' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))

                if distanceType == 'negLogLikelihood':
                    pixLik = pixelLikelihood(testImage, bestImage, variances)
                    plt.imshow(pixLik)
                    plt.colorbar()
                    plt.savefig(numDir + 'pixelLikelihoodPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    plt.clf()

                    disp = cv2.normalize(pixLik, pixLik, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imwrite(numDir + 'pixelLikelihood' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))

                    # fgpost, bgpost = layerPosteriors(testImage, bestImage, layerPrior, variances)
                    #
                    # plt.imshow(fgpost)
                    # plt.colorbar()
                    # plt.savefig(numDir + 'fgPosteriorPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    # plt.clf()
                    #
                    # plt.imshow(bgpost)
                    # plt.colorbar()
                    # plt.savefig(numDir + 'fgPosteriorPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    # plt.clf()
                    #
                    # disp = cv2.normalize(fgpost, fgpost, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    # cv2.imwrite(numDir + 'fgPosterior' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))
                    # disp = cv2.normalize(bgpost, bgpost, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    # cv2.imwrite(numDir + 'bgPosterior' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))

                    # plt.imshow(layerPrior)
                    # plt.colorbar()
                    # plt.savefig(numDir + 'priorPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    # plt.clf()

                cv2.imwrite(numDir + 'bestImage' + "_canny_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png' , bestImageEdges)
                cv2.imwrite(numDir + 'bestImage' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(bestImage*255.0))

                imgEdges = cv2.Canny(numpy.uint8(testImage*255), minThresImage,maxThresImage)
                bwEdges1 = cv2.distanceTransform(~imgEdges, cv2.DIST_L2, 5)
                disp = cv2.normalize(bwEdges1, bwEdges1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(numDir + 'dist_transform' +  '.png', disp)

                plt.plot(azimuths, numpy.array(scores))
                plt.xlabel('Azimuth (degrees)')
                plt.ylabel('Score')
                plt.title(distanceType)
                plt.axvline(x=bestAzimuth, linewidth=2, color='b', label='Minimum score azimuth')
                plt.axvline(x=groundTruthAz, linewidth=2, color='g', label='Ground truth azimuth')
                plt.axvline(x=(bestAzimuth + 180) % 360, linewidth=1, color='b', ls='--', label='Minimum distance azimuth + 180')
                fontP = FontProperties()
                fontP.set_size('small')
                x1,x2,y1,y2 = plt.axis()
                plt.axis((0,360,0,y2))
                # plt.legend()
                plt.savefig(numDir + 'performance.png')
                plt.clf()

    scene.objects.unlink(teapot)

    for backgroundModelOut in backgroundModels:

        modelStr = 'background/'
        if not backgroundModelOut:
            modelStr = 'no_background/'

        for distanceTypeOut in distanceTypes:
            directory = 'aztest/'  + 'teapot' + str(teapotTest)  + '/' + modelStr + distanceTypeOut

            experiment = {'teapot':teapotTest, 'bestAzimuths':bestAzimuths[(backgroundModelOut, distanceTypeOut)], 'performance': performance[(backgroundModelOut, distanceTypeOut)], 'elevations':elevations[(backgroundModelOut, distanceTypeOut)], 'groundTruthAzimuths': groundTruthRelAzimuths[(backgroundModelOut, distanceTypeOut)], 'selTest':selTest, 'expSelTest':expSelTest}
            with open(directory + 'experiment.pickle', 'wb') as pfile:
                pickle.dump(experiment, pfile)

            plt.scatter(elevations[(backgroundModelOut, distanceTypeOut)], performance[(backgroundModelOut, distanceTypeOut)])
            plt.xlabel('Elevation (degrees)')
            plt.ylabel('Angular error')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,90,-180,180))
            plt.title('Performance scatter plot')
            plt.savefig(directory + '_elev-performance-scatter.png')
            plt.clf()

            plt.scatter(occlusions[(backgroundModelOut, distanceTypeOut)]*100.0, performance[(backgroundModelOut, distanceTypeOut)])
            plt.xlabel('Occlusion (%)')
            plt.ylabel('Angular error')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,100,-180,180))
            plt.title('Performance scatter plot')
            plt.savefig(directory + '_occlusion-performance-scatter.png')
            plt.clf()

            plt.scatter(groundTruthAzimuths[(backgroundModelOut, distanceTypeOut)], performance[(backgroundModelOut, distanceTypeOut)])
            plt.xlabel('Azimuth (degrees)')
            plt.ylabel('Angular error')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,360,-180,180))
            plt.title('Performance scatter plot')
            plt.savefig(directory  + '_azimuth-performance-scatter.png')
            plt.clf()

            plt.hist(performance[(backgroundModelOut, distanceTypeOut)], bins=36)
            plt.xlabel('Angular error')
            plt.ylabel('Counts')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((-180,180,0, y2))
            plt.title('Performance histogram')
            plt.savefig(directory  + '_performance-histogram.png')
            plt.clf()
            # experimentFile = 'aztest/teapotsc7549b28656181c91bff71a472da9153Teapot N311012_cleaned.pickle'
            # with open(experimentFile, 'rb') as pfile:
            #     experiment = pickle.load( pfile)

            headers=["Best global fit", ""]
            table = [["Mean angular error", numpy.mean(numpy.abs(performance[(backgroundModelOut, distanceTypeOut)]))],["Median angualar error",numpy.median(numpy.abs(performance[(backgroundModelOut, distanceTypeOut)]))]]
            performanceTable = tabulate(table, tablefmt="latex", floatfmt=".1f")

            with open(directory + 'performance.tex', 'w') as expfile:
                expfile.write(performanceTable)

print("Finished the experiment")


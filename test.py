#!/usr/bin/env python3.4m
 
import sceneimport
from utils import *
from score_image import *
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import cv2

numpy.random.seed(1)

inchToMeter = 0.0254

sceneFile = '../databaseFull/scenes/scene00051.txt'
targetIndex = 9
roomName = 'room09'

cam = bpy.data.cameras.new("MainCamera")
camera = bpy.data.objects.new("MainCamera", cam)

world = bpy.data.worlds.new("MainWorld")

groundTruth, imageFiles, segmentFiles, prefixes = loadGroundTruth()

instances = sceneimport.loadScene(sceneFile)

targetParentPosition = instances[targetIndex][1]

[targetScenes, targetModels] = sceneimport.loadTargetModels()
[blenderScenes, modelInstances] = sceneimport.importBlenderScenes(instances, targetIndex)
width = 110
height = 110

scene = sceneimport.composeScene(modelInstances, targetIndex)

# scene.view_settings.exposure = 1.5
# scene.view_settings.gamma = 1.5

scene.update()
bpy.context.screen.scene = scene

useCycles = False
distance = 0.45
numSamples = 16

setupScene(scene, modelInstances, targetIndex,roomName, world, distance, camera, width, height, numSamples, useCycles)

originalLoc = mathutils.Vector((0,-distance , 0))


# labels = numpy.column_stack((numpy.cos(groundTruthAzs*numpy.pi/180), numpy.sin(groundTruthAzs*numpy.pi/180), numpy.cos(groundTruthAzs*numpy.pi/180.0), numpy.sin(groundTruthAzs*numpy.pi/180.0)))

numpy.random.seed(1)

minThresTemplate = 10
maxThresTemplate = 100
minThresImage = 50
maxThresImage = 150

baseDir = '../databaseFull/models/'

experimentTeapots = [3]
# experimentTeapots = ['teapots/fa1fa0818738e932924ed4f13e49b59d/Teapot N300912','teapots/c7549b28656181c91bff71a472da9153/Teapot N311012', 'teapots/1c43a79bd6d814c84a0fee00d66a5e35/Teapot']

outputExperiments = []

# distanceTypes = ['chamferDataToModel', 'robustChamferDataToModel', 'sqDistImages', 'robustSqDistImages']
distanceTypes = ['sqDistImages', 'negLogLikelihood']
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
vars = numpy.array([])

for backgroundModel in backgroundModels:

    scene.layers[1] = True
    scene.layers[0] = False
    scene.render.layers[0].use = False
    scene.render.layers[1].use = True

    if backgroundModel:
        scene.layers[1] = False
        scene.layers[0] = True
        scene.render.layers[0].use = True
        scene.render.layers[1].use = False

    scene.update()

    computingSqDists = False
    sqDistsSeq = []

    for teapotTest in experimentTeapots:
        robust = True
        robustScale = 0

        teapot = targetModels[teapotTest]
        teapot.layers[0] = True
        teapot.layers[1] = True
        scene.objects.link(teapot)

        scene.update()


        for distanceType in distanceTypes:

            computingSqDists = not computingSqDists
            if not computingSqDists:
                sqRes = numpy.concatenate([aux[..., numpy.newaxis] for aux in sqDistsSeq], axis=-1)
                vars = computeVariances(sqRes)
                vars[numpy.where(vars <= 1)] = 1/255.0
            robust = not robust
            if robust is False:
                robustScale = 0
            experiment = {}
            # ipdb.set_trace()
            indices = numpy.where(groundTruth[:, 3] == teapotTest)

            selTest = indices[0]
            # selTest = numpy.random.permutation(selTest)
            numTests = len(selTest)

            performance = numpy.array([])
            elevations = numpy.array([])
            groundTruthRelAzimuths = numpy.array([])
            groundTruthAzimuths = numpy.array([])
            bestRelAzimuths= numpy.array([])
            bestAzimuths= numpy.array([])
            occlusions = numpy.array([])

            expSelTest = numpy.arange(0,numTests,int(numTests/10))

            for selTestNum in expSelTest:

                teapot.matrix_world = mathutils.Matrix.Translation(targetParentPosition)
                center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
                original_matrix_world = teapot.matrix_world.copy()
                teapot.matrix_world = original_matrix_world

                test = selTest[selTestNum]

                groundTruthAz = groundTruth[test,0]
                groundTruthObjAz = groundTruth[test,1]
                groundTruthRelAz = numpy.arctan2(numpy.sin((groundTruthAz-groundTruthObjAz)*numpy.pi/180), numpy.cos((groundTruthAz-groundTruthObjAz)*numpy.pi/180))*180/numpy.pi
                groundTruthEl = groundTruth[test,2]
                occlusion = groundTruth[test,5]

                azimuthRot = mathutils.Matrix.Rotation(radians(-groundTruthObjAz), 4, 'Z')
                teapot.matrix_world = mathutils.Matrix.Translation(original_matrix_world.to_translation()) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world

                scores = []
                relAzimuths = []
                azimuths = []
                modelStr = 'background/'
                if not backgroundModel:
                    modelStr = 'no_background/'
                directory = 'aztest/'  + 'teapot' + str(teapotTest)  + '/' + modelStr + distanceType
                if not os.path.exists(directory):
                    os.makedirs(directory)


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

                    methodParams = {'vars':vars, 'layerPrior': layerPrior,  'scale': robustScale, 'minThresImage': minThresImage, 'maxThresImage': maxThresImage, 'minThresTemplate': minThresTemplate, 'maxThresTemplate': maxThresTemplate}

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


                if robust is False:
                    robustScale = 1.4826 * numpy.sqrt(numpy.median(scores))

                # error = numpy.arctan2(numpy.sin((groundTruthRelAz-bestRelAzimuth)*numpy.pi/180), numpy.cos((groundTruthRelAz-bestRelAzimuth)*numpy.pi/180))*180/numpy.pi
                error = numpy.arctan2(numpy.sin((groundTruthAz-bestAzimuth)*numpy.pi/180), numpy.cos((groundTruthAz-bestAzimuth)*numpy.pi/180))*180/numpy.pi
                performance = numpy.append(performance, error)
                elevations = numpy.append(elevations, elevation)
                bestAzimuths = numpy.append(bestAzimuths, bestAzimuth)
                groundTruthAzimuths = numpy.append(groundTruthAzimuths, groundTruthAz)
                occlusions = numpy.append(occlusions, occlusion)

                sqDist = sqDistImages(bestImage, testImage)
                disp = cv2.normalize(sqDist, sqDist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(numDir + 'sqDists' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))

                if computingSqDists:
                     sqDistsSeq = sqDistsSeq + [sqDist]

                if distanceType == 'negLogLikelihood':
                    pixLik = pixelLikelihood(testImage, bestImage, layerPrior, vars)
                    plt.imshow(pixLik)
                    plt.colorbar()
                    plt.savefig(numDir + 'pixelLikelihoodPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    plt.clf()

                    disp = cv2.normalize(pixLik, pixLik, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imwrite(numDir + 'pixelLikelihood' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))

                    # fgpost, bgpost = layerPosteriors(testImage, bestImage, layerPrior, vars)
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


            experiment = {'methodParams': methodParams, 'distanceType': distanceType, 'teapot':teapotTest, 'bestAzimuths':bestAzimuths, 'performance': performance, 'elevations':elevations, 'groundTruthAzimuths': groundTruthRelAzimuths, 'selTest':selTest, 'expSelTest':expSelTest}
            outputExperiments.append(experiment)
            with open(directory + 'experiment.pickle', 'wb') as pfile:
                pickle.dump(experiment, pfile)

            plt.scatter(elevations, performance)
            plt.xlabel('Elevation (degrees)')
            plt.ylabel('Angular error')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,90,-180,180))
            plt.title('Performance scatter plot')
            plt.savefig(directory + '_elev-performance-scatter.png')
            plt.clf()

            plt.scatter(groundTruthAzimuths, performance)
            plt.xlabel('Azimuth (degrees)')
            plt.ylabel('Angular error')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,360,-180,180))
            plt.title('Performance scatter plot')
            plt.savefig(directory  + '_azimuth-performance-scatter.png')
            plt.clf()

            plt.hist(performance, bins=36)
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
            table = [["Mean angular error", numpy.mean(numpy.abs(performance))],["Median angualar error",numpy.median(numpy.abs(performance))]]
            performanceTable = tabulate(table, tablefmt="latex", floatfmt=".1f")

            with open(directory + 'performance.tex', 'w') as expfile:
                expfile.write(performanceTable)

        scene.objects.unlink(teapot)
            # Cleanup
            # for obji in scene.objects:
            #     if obji.type == 'MESH':
            #         obji.user_clear()
            #         bpy.data.objects.remove(obji)

            # scene.user_clear()
            # bpy.ops.scene.delete()

print("Finished the experiment")

     


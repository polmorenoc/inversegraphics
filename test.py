#!/usr/bin/env python3.4m
import matplotlib
matplotlib.use('Agg')
import sceneimport
from utils import *
from score_image import *
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import cv2

plt.ioff()

numpy.random.seed(1)

inchToMeter = 0.0254

rendersDir = '../data/output/'
baseTestDir = '../data/aztest/'

groundTruth, imageFiles, segmentFiles,segmentSingleFiles, unoccludedFiles, prefixes = loadGroundTruth(rendersDir)

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
distanceTypes = ['negLogLikelihood']
# distanceTypes = ['negLogLikelihood','negLogLikelihoodRobust']
masks = numpy.array([])
segmentImages = []
for segment in segmentSingleFiles:
    print(segment)
    segmentImg = cv2.imread(segment, cv2.IMREAD_ANYDEPTH)/255.0
    segmentImg = segmentImg[..., numpy.newaxis]
    segmentImages = segmentImages + [segmentImg]

masks = numpy.concatenate([aux for aux in segmentImages], axis=-1)

layerPrior = globalLayerPrior(masks)

backgroundModels = ['UNOCCLUDED']
# backgroundModels = ['SINGLE', 'UNOCCLUDED']

completeScene = True
if 'FULL' not in backgroundModels:
    completeScene = False

# sortedScenesIndices = numpy.argsort(groundTruth[:,6])
# sortedGroundTruth = groundTruth[sortedScenesIndices, :]
# sortedMasks = masks[:,:,sortedScenesIndices]
# sortedImageFiles = imageFiles[sortedScenesIndices]

filterScenes = [19]

indicesOccluded = (groundTruth[:,5] < 0.90) & (groundTruth[:,5] > 0.05)

indicesScenes = (groundTruth[:,6] != 19)

partsOccluded = (groundTruth[:,14] == 1) | (groundTruth[:,13] == 1)

indicesPrefix = [ x==y for (x,y) in zip(prefixes, ['_occluded']*len(prefixes))]

[targetScenes, targetModels, transformations] = sceneimport.loadTargetModels(experimentTeapots)

numExperiments = 100

spout = mathutils.Vector((-6.2, -0.16, 6.25))
handle = mathutils.Vector((6.2, 0.2, 5.7))
tip = mathutils.Vector((0, 0, 8))

for teapotIdx, teapotTest in enumerate(experimentTeapots):

    teapot = targetModels[teapotIdx]
    teapot.layers[0] = True
    teapot.layers[1] = True

    transformation = transformations[teapotIdx]
    spout = transformation * spout
    handle = transformation * handle
    tip = transformation * tip

    print("Experiment on teapot " + teapot.name)

    indicesTeapot = (groundTruth[:, 3] == teapotTest)

    indices = numpy.where(indicesTeapot & indicesOccluded & indicesScenes & partsOccluded )

    selTest = indices[0]

    numTests = len(selTest)
    print ("Total of " + str(numTests) + " tests")

    # expSelTest = [0,1]
    numExperiments = min(numTests, numExperiments)
    expSelTest = numpy.array(numpy.arange(0,numTests-1,int(numTests-1)/numExperiments), dtype=numpy.int)

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

        testResults = {}
        testAzimuths = {}

        groundTruthAz = groundTruth[test,0]
        groundTruthObjAz = groundTruth[test,1]
        groundTruthRelAz = numpy.arctan2(numpy.sin((groundTruthAz-groundTruthObjAz)*numpy.pi/180), numpy.cos((groundTruthAz-groundTruthObjAz)*numpy.pi/180))*180/numpy.pi
        groundTruthEl = groundTruth[test,2]
        occlusion = groundTruth[test,5]
        sceneNum = int(groundTruth[test,6])
        targetIndex= int(groundTruth[test,7])

        sampleDir = baseTestDir + 'teapot' + str(teapotTest)  + '/' 'test_samples/num' + str(test) + '_azim' + str(int(groundTruthAz)) + '_elev' + str(int(groundTruthEl)) + '_occlusion' + str(occlusion) + '/'
        if not os.path.exists(sampleDir):
            os.makedirs(sampleDir)

        if currentScene != sceneNum:
            if currentScene != -1:
                # Cleanup
                for objnum, obji in enumerate(scene.objects):
                    if obji.name != teapot.name and obji.type == 'EMPTY' and obji.dupli_type == 'GROUP':
                        deleteInstance(obji)
                    elif obji.type == 'LAMP':
                        obji.data.user_clear()
                        bpy.data.lamps.remove(obji.data)
                        obji.user_clear()
                        bpy.data.objects.remove(obji)
                    elif obji.name != teapot.name:
                        obji.user_clear()
                        bpy.data.objects.remove(obji)

                camera.user_clear()
                bpy.data.objects.remove(camera)
                cam.user_clear()
                bpy.data.cameras.remove(cam)
                world.user_clear()
                bpy.data.worlds.remove(world)

                bpy.context.screen.scene = bpy.data.scenes['Scene']
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

            # bpy.context.scene.use_nodes = True
            # tree = bpy.context.scene.node_tree

            # rl = tree.nodes[0]
            # links = tree.links
            # v = tree.nodes.new('CompositorNodeViewer')
            # v.location = 750,210
            # v.use_alpha = False
            # links.new(rl.outputs[0], v.inputs[0])  # link Image output to Viewer input

            scene.update()
            scene.render.threads = 8
            scene.render.threads_mode = 'AUTO'
            bpy.context.screen.scene = scene

            cycles = bpy.context.scene.cycles
            scene.render.tile_x = 55
            scene.render.tile_y = 55

            cam = bpy.data.cameras.new("MainCamera")
            camera = bpy.data.objects.new("MainCamera", cam)
            world = bpy.data.worlds.new("MainWorld")

            setupScene(scene, targetIndex, roomName, world, distance, camera, width, height, numSamples, useCycles, useGPU)

            scene.objects.link(teapot)
            teapot.layers[1] = True
            scene.update()

            teapot.matrix_world = mathutils.Matrix.Translation(targetParentPosition)
            center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
            original_matrix_world = teapot.matrix_world.copy()
            teapot.matrix_world = original_matrix_world

            # bpy.ops.mesh.primitive_uv_sphere_add(size=0.005, location=spout)
            # sphereSpout = scene.objects[0]
            # bpy.ops.mesh.primitive_uv_sphere_add(size=0.005, location=handle)
            # sphereHandle = scene.objects[0]
            # bpy.ops.mesh.primitive_uv_sphere_add(size=0.005, location=tip)
            # sphereTip = scene.objects[0]
            # sphereTip.layers[1] = True
            # sphereSpout.layers[1] = True
            # sphereHandle.layers[1] = True

        azimuthRot = mathutils.Matrix.Rotation(radians(-groundTruthObjAz), 4, 'Z')
        groundTruthTransf = mathutils.Matrix.Translation(original_matrix_world.to_translation()) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation()))
        teapot.matrix_world = groundTruthTransf * original_matrix_world

        sphereSpout_matrix_world = teapot.matrix_world * mathutils.Matrix.Translation(spout)
        sphereHandle_matrix_world = teapot.matrix_world * mathutils.Matrix.Translation(handle)
        sphereTip_matrix_world = teapot.matrix_world * mathutils.Matrix.Translation(tip)

        for backgroundModel in backgroundModels:

            variances = numpy.ones([height, width, 3])*2/255.0

            print("Experiment on background model " + backgroundModel)

            scene.layers[1] = True
            scene.layers[0] = False
            scene.render.layers[0].use = False
            scene.render.layers[1].use = True
            teapot.layers[1] = True

            if backgroundModel == 'FULL':
                scene.layers[1] = False
                scene.layers[0] = True
                scene.render.layers[0].use = True
                scene.render.layers[1].use = False
                teapot.layers[0] = True

            scene.update()

            sqDistsSeq = []

            for distIdx, distanceType in enumerate(distanceTypes):
                print("Experiment on model " + distanceType)

                computingSqDists = False

                if distIdx == 0:
                    computingSqDists = True

                if not computingSqDists and not backgroundModel == 'FULL':
                    sqRes = numpy.concatenate([aux[..., numpy.newaxis] for aux in sqDistsSeq], axis=-1)
                    variances = computeVariances(sqRes)
                    variances[numpy.where(variances <= 1)] = 2.0/255.0

                # robust = not robust
                # if robust is False:
                #     robustScale = 0

                scores = []
                relAzimuths = []
                azimuths = []

                directory =  baseTestDir + 'teapot' + str(teapotTest)  + '/' + backgroundModel + '_' + distanceType
                if not os.path.exists('../data/aztest/'  + 'teapot' + str(teapotTest)  + '/'):
                    os.makedirs('../data/aztest/'  + 'teapot' + str(teapotTest)  + '/')

                if not os.path.exists(directory + 'test_samples'):
                    os.makedirs(directory + 'test_samples')

                numDir = directory +  'test_samples/num' + str(test) + '_azim' + str(int(groundTruthAz)) + '_elev' + str(int(groundTruthEl)) + '_occlusion' + str(occlusion) + '/'
                if not os.path.exists(numDir):
                    os.makedirs(numDir)

                testImage = cv2.imread(imageFiles[test])/255.0
                if backgroundModel == 'UNOCCLUDED':
                    testImage = cv2.imread(unoccludedFiles[test])/255.0
                # testImage = cv2.cvtColor(numpy.float32(rgbTestImage*255), cv2.COLOR_RGB2BGR)/255.0

                testMask = masks[:,:,test]

                testImageEdges = cv2.Canny(numpy.uint8(testImage*255), minThresImage,maxThresImage)
                cv2.imwrite(numDir + "image_canny_" + backgroundModel +  ".png" , testImageEdges)
                cv2.imwrite(numDir + "image_" + backgroundModel + ".png" , numpy.uint8(testImage*255))

                cv2.imwrite(sampleDir + "image_canny_" + backgroundModel +  ".png" , testImageEdges)
                cv2.imwrite(sampleDir + "image_" + backgroundModel + ".png" , numpy.uint8(testImage*255))

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

                    # projMat = projection_matrix(scene.camera.data, scene)
                    # ipdb.set_trace()

                    spoutlocation = image_project(scene, camera, sphereSpout_matrix_world.to_translation())
                    handlelocation = image_project(scene, camera,sphereHandle_matrix_world.to_translation())
                    spherelocation = image_project(scene, camera, sphereTip_matrix_world.to_translation())

                    # touched = closestCameraIntersection(scene, sphereSpout.matrix_world.to_translation())

                    # scene.objects.unlink(sphereSpout)
                    # scene.objects.unlink(sphereHandle)
                    # scene.objects.unlink(sphereTip)
                    result, object, matrix, location, normal = scene.ray_cast(scene.camera.location, sphereSpout_matrix_world.to_translation())

                    scene.render.filepath = numDir  + '_blender.png'

                    bpy.ops.render.render( write_still=True )

                    # image2 = numpy.flipud(numpy.array(bpy.data.images['Viewer Node'].extract_render(scene=scene)).reshape([256,256,4]))[:,:,0:3]

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

                    methodParams = {'backgroundModel': backgroundModel, 'testMask': testMask, 'variances':variances, 'layerPrior': layerPrior, 'minThresImage': minThresImage, 'maxThresImage': maxThresImage, 'minThresTemplate': minThresTemplate, 'maxThresTemplate': maxThresTemplate}

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

                testResults[(backgroundModel, distanceType)] = scores
                testAzimuths[(backgroundModel, distanceType)] = azimuths

                sqDist = sqDistImages(bestImage, testImage)
                disp = cv2.normalize(sqDist, sqDist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(numDir + 'sqDists' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))

                if computingSqDists:
                    sqDistsSeq = sqDistsSeq + [sqDist]

                if distanceType == 'negLogLikelihoodRobust':
                    pixLik = pixelLikelihoodRobust(testImage, bestImage, testMask, backgroundModel, layerPrior, variances)
                    fig = plt.figure()
                    plt.imshow(pixLik)
                    plt.colorbar()
                    fig.savefig(numDir + 'pixelLikelihoodRobustPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    plt.close(fig)

                    disp = cv2.normalize(pixLik, pixLik, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imwrite(numDir + 'pixelLikelihoodRobust' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))

                    fgpost, bgpost = layerPosteriorsRobust(testImage, bestImage, testMask, backgroundModel, layerPrior, variances)

                    # plt.imshow(testImage)
                    # plt.show()
                    # plt.imshow(bestImage)
                    # plt.show()

                    # z = fgpost + bgpost
                                                                                                                                                                                                                                                     # plt.imshow(z)
                    # plt.show()
                    # plt.imshow(fgpost)
                    # plt.show()
                    # plt.imshow(bgpost)
                    # plt.show()
                    # ipdb.set_trace
                    assert(numpy.abs(numpy.sum(fgpost + bgpost - testMask)) < 0.01)
                    fig = plt.figure()
                    plt.imshow(fgpost)
                    plt.colorbar()
                    fig.savefig(numDir + 'fgPosteriorPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    plt.close(fig)

                    fig = plt.figure()
                    plt.imshow(bgpost)
                    plt.colorbar()
                    fig.savefig(numDir + 'bgPosteriorPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    plt.close(fig)

                    disp = cv2.normalize(fgpost, fgpost, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imwrite(numDir + 'fgPosterior' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))
                    disp = cv2.normalize(bgpost, bgpost, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imwrite(numDir + 'bgPosterior' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))

                if distanceType == 'negLogLikelihood':
                    pixLik = pixelLikelihood(testImage, bestImage, testMask, backgroundModel, variances)
                    fig = plt.figure()
                    plt.imshow(pixLik)
                    plt.colorbar()
                    fig.savefig(numDir + 'pixelLikelihoodPlot' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png')
                    plt.close(fig)

                    disp = cv2.normalize(pixLik, pixLik, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imwrite(numDir + 'pixelLikelihood' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(disp))


                cv2.imwrite(numDir + 'bestImage' + "_canny_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png' , bestImageEdges)
                cv2.imwrite(numDir + 'bestImage' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(bestImage*255.0))

                imgEdges = cv2.Canny(numpy.uint8(testImage*255), minThresImage,maxThresImage)
                bwEdges1 = cv2.distanceTransform(~imgEdges, cv2.DIST_L2, 5)
                disp = cv2.normalize(bwEdges1, bwEdges1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(numDir + 'dist_transform' +  '.png', disp)

                # ipdb.set_trace()
                fig = plt.figure()
                plt.plot(azimuths, numpy.array(scores))
                plt.xlabel('Azimuth (degrees)')
                plt.ylabel('Negative Log Likelihood')
                plt.title(distanceType)
                plt.axvline(x=bestAzimuth, linewidth=2, color='b', label='Minimum score azimuth')
                plt.axvline(x=groundTruthAz, linewidth=2, color='g', label='Ground truth azimuth')
                plt.axvline(x=(bestAzimuth + 180) % 360, linewidth=1, color='b', ls='--', label='Minimum distance azimuth + 180')
                fontP = FontProperties()
                fontP.set_size('small')
                x1,x2,y1,y2 = plt.axis()
                plt.axis((0,360,y1,y2))

                # plt.legend()
                fig.savefig(numDir + 'performance.png')
                plt.close(fig)
        fig = plt.figure()
        backgroundModelOut = 'UNOCCLUDED'
        distanceTypeOut = 'negLogLikelihood'
        maxY = numpy.max(numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]))
        minY = numpy.min(numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]))
        plt.plot(testAzimuths[(backgroundModelOut, distanceTypeOut)], (numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]) - minY)/ (maxY-minY), color='g', label='Clean model')
        plt.axvline(x=testAzimuths[(backgroundModelOut, distanceTypeOut)][numpy.argmin(numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]))], linewidth=1, color='g', ls='--')


        backgroundModelOut = 'SINGLE'
        distanceTypeOut = 'negLogLikelihood'
        maxY = numpy.max(numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]))
        minY = numpy.min(numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]))
        plt.plot(testAzimuths[(backgroundModelOut, distanceTypeOut)], (numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]) - minY)/ (maxY-minY), color='r', label='Normal model')
        plt.axvline(x=testAzimuths[(backgroundModelOut, distanceTypeOut)][numpy.argmin(numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]))], linewidth=1, color='r', ls='--')

        backgroundModelOut = 'SINGLE'
        distanceTypeOut = 'negLogLikelihoodRobust'
        maxY = numpy.max(numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]))
        minY = numpy.min(numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]))
        plt.plot(testAzimuths[(backgroundModelOut, distanceTypeOut)], (numpy.array(testResults[(backgroundModelOut, distanceTypeOut)])- minY) / (maxY-minY), color='c', label='Robust model')
        plt.axvline(x=testAzimuths[(backgroundModelOut, distanceTypeOut)][numpy.argmin(numpy.array(testResults[(backgroundModelOut, distanceTypeOut)]))], linewidth=1, color='c', ls='--')

        plt.xlabel('Azimuth (degrees)')
        plt.ylabel('Negative Log Likelihood')
        fontP = FontProperties()
        fontP.set_size('small')
        # plt.axvline(x=bestAzimuth, linewidth=2, color='r', label='Minimum score azimuth')
        plt.axvline(x=groundTruthAz, linewidth=1, color='b', label='GT azimuth')
        # plt.axvline(x=(bestAzimuth + 180) % 360, linewidth=1, color='r', ls='--', label='Minimum distance azimuth + 180')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((0,360,y1,y2))
        lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

        fig.savefig(sampleDir + 'performance.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)

    scene.objects.unlink(teapot)

    for backgroundModelOut in backgroundModels:

        for distanceTypeOut in distanceTypes:
            directory = baseTestDir  + 'teapot' + str(teapotTest)  + '/' + backgroundModelOut + distanceTypeOut

            experiment = {'distanceType':distanceTypeOut, 'backgroundModel':backgroundModelOut, 'rendersDir':rendersDir, 'scores':scores, 'azimuths':azimuths,'teapot':teapotTest, 'bestAzimuths':bestAzimuths[(backgroundModelOut, distanceTypeOut)], 'performance': performance[(backgroundModelOut, distanceTypeOut)], 'elevations':elevations[(backgroundModelOut, distanceTypeOut)], 'groundTruthAzimuths': groundTruthRelAzimuths[(backgroundModelOut, distanceTypeOut)], 'occlusions': occlusions[(backgroundModelOut, distanceTypeOut)],'selTest':selTest, 'test':test}
            with open(directory + 'experiment.pickle', 'wb') as pfile:
                pickle.dump(experiment, pfile)
            fig = plt.figure()
            plt.scatter(elevations[(backgroundModelOut, distanceTypeOut)], performance[(backgroundModelOut, distanceTypeOut)])
            plt.xlabel('Elevation (degrees)')
            plt.ylabel('Angular error')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,90,-180,180))
            plt.title('Performance scatter plot')
            fig.savefig(directory + '_elev-performance-scatter.png')
            plt.close(fig)

            fig = plt.figure()
            plt.scatter(occlusions[(backgroundModelOut, distanceTypeOut)]*100.0, performance[(backgroundModelOut, distanceTypeOut)])
            plt.xlabel('Occlusion (%)')
            plt.ylabel('Angular error')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,100,-180,180))
            plt.title('Performance scatter plot')
            fig.savefig(directory + '_occlusion-performance-scatter.png')
            plt.close(fig)

            fig = plt.figure()
            plt.scatter(groundTruthAzimuths[(backgroundModelOut, distanceTypeOut)], performance[(backgroundModelOut, distanceTypeOut)])
            plt.xlabel('Azimuth (degrees)')
            plt.ylabel('Angular error')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,360,-180,180))
            plt.title('Performance scatter plot')
            fig.savefig(directory  + '_azimuth-performance-scatter.png')
            plt.close(fig)

            fig = plt.figure()
            plt.hist(performance[(backgroundModelOut, distanceTypeOut)], bins=36)
            plt.xlabel('Angular error')
            plt.ylabel('Counts')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((-180,180,y1, y2))
            plt.title('Performance histogram')
            fig.savefig(directory  + '_performance-histogram.png')
            plt.close(fig)
            # experimentFile = 'aztest/teapotsc7549b28656181c91bff71a472da9153Teapot N311012_cleaned.pickle'
            # with open(experimentFile, 'rb') as pfile:
            #     experiment = pickle.load( pfile)

            headers=["Best global fit", ""]
            table = [["Mean angular error", numpy.mean(numpy.abs(performance[(backgroundModelOut, distanceTypeOut)]))],["Median angualar error",numpy.median(numpy.abs(performance[(backgroundModelOut, distanceTypeOut)]))]]
            performanceTable = tabulate(table, tablefmt="latex", floatfmt=".1f")

            with open(directory + 'performance.tex', 'w') as expfile:
                expfile.imsaveperformanceTable)

    experiment = {'distanceTypes':distanceTypes, 'backgroundModels':backgroundModels, 'rendersDir':rendersDir,'teapot':teapotTest,'testResults':testResults,'testAzimuths':testAzimuths, 'bestAzimuths':bestAzimuths, 'performance': performance, 'elevations':elevations, 'groundTruthAzimuths': groundTruthRelAzimuths, 'occlusions': occlusions}
    with open(baseTestDir  + 'teapot' + str(teapotTest)  + '/' + 'experiment.pickle', 'wb') as pfile:
        pickle.dump(experiment, pfile)

print("Finished the experiment")


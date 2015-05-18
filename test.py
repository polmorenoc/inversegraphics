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

sceneFile = '../databaseFull/scenes/scene00051_2.txt'
targetIndex = 2
roomName = 'room09'

world = bpy.context.scene.world

instances = sceneimport.loadScene(sceneFile)

targetParentPosition = instances[targetIndex][1]

[targetScenes, targetModels] = sceneimport.loadTargetModels()
[blenderScenes, modelInstances] = sceneimport.importBlenderScenes(instances, targetIndex)
width = 100
height = 100
camera = bpy.data.scenes['Scene'].objects[2]
scene = sceneimport.composeScene(modelInstances, targetIndex)

scene.update()
bpy.context.screen.scene = scene

useCycles = False
distance = 0.4
numSamples = 1024

setupScene(scene, modelInstances, targetIndex,roomName, world, distance, camera, width, height, numSamples, useCycles)

originalLoc = mathutils.Vector((0,-distance , 0))
groundTruth, imageFiles = loadGroundTruth()

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
distanceTypes = ['chamferDataToModel', 'ignoreSqDistImages', 'sqDistImages', 'chamferModelToData']


for teapotTest in experimentTeapots:
    robust = True
    robustScale = 0

    for distanceType in distanceTypes:
        robust = ~robust
        if robust is False:
            robustScale = 0
        

        experiment = {}
        # ipdb.set_trace()
        indices = numpy.where(groundTruth[:, 3] == teapotTest)

        selTest = indices[0]
        selTest = numpy.random.permutation(selTest)
        numTests = len(selTest)


        teapot = targetModels[teapotTest]
        teapot.matrix_world = mathutils.Matrix.Translation(targetParentPosition)
        center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
        original_matrix_world = teapot.matrix_world.copy()
        
        azimuthRot = mathutils.Matrix.Rotation(radians(0), 4, 'Z')
        teapot.matrix_world = mathutils.Matrix.Translation(original_matrix_world.to_translation()) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world
        scene.objects.link(teapot)

        performance = numpy.array([])
        elevations = numpy.array([]) 
        groundTruthRelAzimuths = numpy.array([])
        bestRelAzimuths= numpy.array([]) 


        expSelTest = numpy.arange(0,numTests,int(numTests/100))

        for selTestNum in expSelTest:

            test = selTest[selTestNum]

            groundTruthAz = groundTruth[test,0]
            groundTruthObjAz = groundTruth[test,1]
            groundTruthRelAz = numpy.arctan2(numpy.sin((groundTruthAz-groundTruthObjAz)*numpy.pi/180), numpy.cos((groundTruthAz-groundTruthObjAz)*numpy.pi/180))*180/numpy.pi
            groundTruthEl = groundTruth[test,2]

            scores = []
            relAzimuths = []  
            directory = 'aztest/'  + 'teapot' + str(teapotTest)  + '/' + distanceType
            if not os.path.exists(directory):
                os.makedirs(directory)


            if not os.path.exists(directory + 'test_samples'):
                os.makedirs(directory + 'test_samples')


            numDir = directory +  'test_samples/num' + str(test) + '_azim' + str(int(groundTruthRelAz)) + '_elev' + str(int(groundTruthEl)) + '/'
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
                
                scene.render.filepath = directory  + "_blender_" +  str(test) +  "_az" + '%.1f' % azimuth + '_dist' + '%.1f' % distance + '.png'
                
                bpy.ops.render.render( write_still=True )

                # image = cv2.imread(scene.render.filepath, cv2.IMREAD_ANYDEPTH)

                blendImage = bpy.data.images['Render Result']

                image = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height,width,4]))
                # ipdb.set_trace()
                # Truncate intensities larger than 1.
                # image[numpy.where(image > 1)] = 1
                # ipdb.set_trace()
                # image[0:20, 75:100, :] = 0

                image = cv2.cvtColor(numpy.float32(image*255), cv2.COLOR_RGB2BGR)

                cv2.imshow('ImageWindow',image)

                cv2.waitKey()

                methodParams = {'scale': robustScale, 'minThresImage': minThresImage, 'maxThresImage': maxThresImage, 'minThresTemplate': minThresTemplate, 'maxThresTemplate': maxThresTemplate}
                

                distance = scoreImage(testImage, image, distanceType, methodParams)
                cv2.imwrite(numDir + 'image' + "_az" + '%.1f' % azimuth + '_dist' + '%.1f' % distance + '.png', numpy.uint8(image*255.0))

                if distance <= score:
                    imageEdges = cv2.Canny(numpy.uint8(image*255.0), minThresTemplate,maxThresTemplate)
                    bestImageEdges = imageEdges
                    bestImage = image
                    score = distance

                scores.append(distance)
                relAzimuth = numpy.arctan2(numpy.sin((azimuth-groundTruthObjAz)*numpy.pi/180), numpy.cos((azimuth-groundTruthObjAz)*numpy.pi/180))*180/numpy.pi
                relAzimuths.append(relAzimuth)

                                
            bestRelAzimuth = relAzimuths[numpy.argmin(scores)]
            if robust is False:
                robustScale = 1.4826 * numpy.sqrt(numpy.median(scores))

            error = numpy.arctan2(numpy.sin((groundTruthRelAz-bestRelAzimuth)*numpy.pi/180), numpy.cos((groundTruthRelAz-bestRelAzimuth)*numpy.pi/180))*180/numpy.pi
            performance = numpy.append(performance, error)
            elevations = numpy.append(elevations, elevation)
            bestRelAzimuths = numpy.append(bestRelAzimuths, bestRelAzimuth)
            groundTruthRelAzimuths = numpy.append(groundTruthRelAzimuths, groundTruthRelAz)

            cv2.imwrite(numDir + 'bestImage' + "_canny_az" + '%.1f' % bestRelAzimuth + '_dist' + '%.1f' % score + '.png' , bestImageEdges)
            cv2.imwrite(numDir + 'bestImage' + "_az" + '%.1f' % bestRelAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(bestImage*255.0))

            imgEdges = cv2.Canny(numpy.uint8(testImage*255), minThresImage,maxThresImage)
            bwEdges1 = cv2.distanceTransform(~imgEdges, cv2.DIST_L2, 5)
            disp = cv2.normalize(bwEdges1, bwEdges1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(numDir + 'dist_transform' +  '.png', disp)

            plt.plot(relAzimuths, numpy.array(scores))
            plt.xlabel('Azimuth (degrees)')
            plt.ylabel('Distance')
            plt.title('Chamfer distance')
            plt.axvline(x=bestRelAzimuth, linewidth=2, color='b', label='Minimum distance azimuth')
            plt.axvline(x=groundTruthRelAz, linewidth=2, color='g', label='Ground truth azimuth')
            plt.axvline(x=(bestRelAzimuth + 180) % 360, linewidth=1, color='b', ls='--', label='Minimum distance azimuth + 180')
            fontP = FontProperties()
            fontP.set_size('small')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,360,0,y2))
            # plt.legend()
            plt.savefig(numDir + 'performance.png')
            plt.clf()


        experiment = {'methodParams': methodParams, 'distanceType': distanceType, 'teapot':teapotTest, 'bestRelAzimuths':bestRelAzimuths, 'performance': performance, 'elevations':elevations, 'groundTruthRelAzimuths': groundTruthRelAzimuths, 'selTest':selTest, 'expSelTest':expSelTest}
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

        plt.scatter(groundTruthRelAzimuths, performance)
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

        # Cleanup
        # for obji in scene.objects:
        #     if obji.type == 'MESH':
        #         obji.user_clear()
        #         bpy.data.objects.remove(obji)

        # scene.user_clear()
        # bpy.ops.scene.delete()   

print("Finished the experiment")

     


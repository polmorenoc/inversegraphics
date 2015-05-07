#!/usr/bin/env python3.4m
 
from utils import *

# bpy.ops.render.render( write_still=True )


lines = [line.strip() for line in open('teapots.txt')]

# lamp = bpy.data.scenes['Scene'].objects[1]
# lamp.location = (0,0.0,1.5)

lamp_data = bpy.data.lamps.new(name="LampTopData", type='AREA')
lamp = bpy.data.objects.new(name="LampTop", object_data=lamp_data)
lamp.location = (0,0.0,2)
lamp.data.energy = 0.004
lamp.data.size = 0.5
lamp.data.use_diffuse = True
# lamp.data.use_nodes = True


lamp_data2 = bpy.data.lamps.new(name="LampBotData", type='POINT')
lamp2 = bpy.data.objects.new(name="LampBot", object_data=lamp_data2)
lamp2.location = (0,0.0,-1.0)
lamp2.data.energy = 0.2
# lamp.data.size = 0.25
lamp2.data.use_diffuse = True
lamp2.data.use_specular = False
# lamp2.data.use_nodes = True


camera = bpy.data.scenes['Scene'].objects[2]

camera.data.angle = 60 * 180 / numpy.pi

distance = 0.5
originalLoc = mathutils.Vector((0,-distance,0.0))

elevation = 0.0
azimuth = 0.0
elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
location = azimuthRot * elevationRot * originalLoc
camera.location = location

look_at(camera, mathutils.Vector((0,0,0)))

world = bpy.context.scene.world

# Environment lighting
world.light_settings.use_environment_light = True
world.light_settings.environment_energy = 0.15
world.horizon_color = mathutils.Color((0.0,0.0,0.0))

width = 230
height = 230

data, images, experiments = loadData()

groundTruthEls = data['altitudes'][0][0][0]
groundTruthAzs = data['azimuths'][0][0][0]

filenames = [name[0] for name in data['filenames'][0][0][0][:]]
ids = [name[0] for name in data['ids'][0][0][0][:]]

labels = numpy.column_stack((numpy.cos(groundTruthAzs*numpy.pi/180), numpy.sin(groundTruthAzs*numpy.pi/180), numpy.cos(groundTruthAzs*numpy.pi/180.0), numpy.sin(groundTruthAzs*numpy.pi/180.0)))

output = scipy.io.loadmat('../data/crossval6div2-hog8-alldataexperiments.mat')['output_data']

numpy.random.seed(1)

minThresTemplate = 10
maxThresTemplate = 100
minThresImage = 50
maxThresImage = 150

baseDir = '../databaseFull/models/'

experimentTeapots = ['teapots/fa1fa0818738e932924ed4f13e49b59d/Teapot N300912','teapots/c7549b28656181c91bff71a472da9153/Teapot N311012', 'teapots/1c43a79bd6d814c84a0fee00d66a5e35/Teapot', 'teapots/a7fa82f5982edfd033da2d90df7af046/Teapot_fixed', 'teapots/8e6a162e707ecdf323c90f8b869f2ce9/Teapot N280912', 'teapots/12b81ec72a967dc1714fc48a3b0c961a/Teapot N260113_fixed']
experimentTeapots = ['teapots/fa1fa0818738e932924ed4f13e49b59d/Teapot N300912','teapots/c7549b28656181c91bff71a472da9153/Teapot N311012', 'teapots/1c43a79bd6d814c84a0fee00d66a5e35/Teapot']

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

        indices = [i for i, s in enumerate(ids) if teapotTest in s]

        selTest = indices
        selTest = numpy.random.permutation(selTest)
        numTests = len(selTest)


        teapot = teapotTest  + '_cleaned'
        fullTeapot = baseDir + teapot

        print("Reading " + fullTeapot + '.dae')

        bpy.ops.scene.new()
        bpy.context.scene.name = teapot
        scene = bpy.context.scene
        # bpy.context.scene.render.engine = 'CYCLES'
        # bpy.context.scene.cycles.samples = 128

        scene.camera = camera

        scene.render.resolution_x = width #perhaps set resolution in code
        scene.render.resolution_y = height
        scene.world = world

        scene.render.filepath = teapot + '.png'

        bpy.utils.collada_import(fullTeapot + '.dae')

        modifySpecular(scene, 0.3)

        # ipdb.set_trace()

        minZ, maxZ = modelHeight(scene)

        minY, maxY = modelWidth(scene)

        scaleZ = 0.254/(maxZ-minZ)
        scaleY = 0.1778/(maxY-minY)

        scale = min(scaleZ, scaleY)

        for mesh in scene.objects:
            if mesh.type == 'MESH':
                scaleMat = mathutils.Matrix.Scale(scale, 4)
                mesh.matrix_world =  scaleMat * mesh.matrix_world
                     
        minZ, maxZ = modelHeight(scene)
        scene.objects.link(lamp2)

        scene.objects.link(lamp)
        
        # lamp2.location = (0,0, 2)


        center = centerOfGeometry(scene)
        for mesh in scene.objects:
            if mesh.type == 'MESH':
                mesh.matrix_world = mathutils.Matrix.Translation(-center) * mesh.matrix_world

        #Rotate the object to the azimuth angle we define as 0.
        rot = mathutils.Matrix.Rotation(radians(90), 4, 'Z')
        rotateMatrixWorld(scene, rot)
        scene.update()

        camera.data.angle = 60 * 180 / numpy.pi


        performance = numpy.array([])
        elevations = numpy.array([]) 
        groundTruthAzimuths = numpy.array([])
        bestAzimuths= numpy.array([]) 


        expSelTest = numpy.arange(0,numTests,int(numTests/30))


        for selTestNum in expSelTest:

            test = selTest[selTestNum]
            groundTruthAz = groundTruthAzs[test]
            groundTruthEl = groundTruthEls[test]
            scores = []
            azimuths = []  
            directory = 'aztest/'  + '_' + teapot.replace("/", "")  + '/' + distanceType
            if not os.path.exists(directory):
                os.makedirs(directory)


            if not os.path.exists(directory + 'test_samples'):
                os.makedirs(directory + 'test_samples')


            numDir = directory +  'test_samples/num' + str(test) + '_azim' + str(int(groundTruthAz)) + '_elev' + str(int(groundTruthEl)) + '/'
            if not os.path.exists(numDir):
                os.makedirs(numDir)


            rgbTestImage = numpy.transpose(images["images"][test])
            testImage = cv2.cvtColor(numpy.float32(rgbTestImage*255), cv2.COLOR_RGB2BGR)/255.0

            testImageEdges = cv2.Canny(numpy.uint8(testImage*255), minThresImage,maxThresImage)
            cv2.imwrite(numDir + "image_canny" + ".png" , testImageEdges)
            cv2.imwrite(numDir + "image" + ".png" , numpy.uint8(testImage*255))

            score = numpy.finfo(numpy.float64).max
                


            elevation = groundTruthEls[test]
            # elevation = -45
            azimuth = 0
            center = centerOfGeometry(scene)
            elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
            # azimuthRot = mathutils.Matrix.Rotation(radians(azimuth), 4, 'Z')
            # location = azimuthRot * elevationRot * (center + originalLoc)
            # camera.location = location
            # scene.update()
            # look_at(camera, center)
            # scene.update()

            # bpy.ops.render.render( write_still=False )   

            # blendImage = bpy.data.images['Render Result']
            # image = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height/2,width/2,4]))
            # image[numpy.where(image > 1)] = 1

            # distance = getChamferDistance(testImage, image, minThresImage, maxThresImage, minThresTemplate, maxThresTemplate)

            for azimuth in numpy.arange(0,360,5):

                azimuthRot = mathutils.Matrix.Rotation(radians(azimuth), 4, 'Z')
                location = azimuthRot * elevationRot * (center + originalLoc)
                camera.location = location
                scene.update()
                look_at(camera, center)          
                scene.update()
                

                scene.render.filepath = directory  +  teapot.replace("/", "") +  "blender_" + '_' +  str(test) +  "_az" + '%.1f' % azimuth + '_dist' + '%.1f' % distance + '.png'
                
                bpy.ops.render.render( write_still=False )

                # image = cv2.imread(scene.render.filepath, cv2.IMREAD_ANYDEPTH)

                blendImage = bpy.data.images['Render Result']

                image = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height/2,width/2,4]))[7:107,7:107,0:3]

                # Truncate intensities larger than 1.
                image[numpy.where(image > 1)] = 1
                # ipdb.set_trace()
                image[0:20, 75:100, :] = 0

                image = cv2.cvtColor(numpy.float32(image*255), cv2.COLOR_RGB2BGR)/255.0

                methodParams = {'scale': robustScale, 'minThresImage': minThresImage, 'maxThresImage': maxThresImage, 'minThresTemplate': minThresTemplate, 'maxThresTemplate': maxThresTemplate}
                

                distance = scoreImage(testImage, image, distanceType, methodParams)
                cv2.imwrite(numDir + 'image' + "_az" + '%.1f' % azimuth + '_dist' + '%.1f' % distance + '.png', numpy.uint8(image*255.0))

                if distance <= score:
                    imageEdges = cv2.Canny(numpy.uint8(image*255.0), minThresTemplate,maxThresTemplate)
                    bestImageEdges = imageEdges
                    bestImage = image
                    score = distance


                scores.append(distance)
                azimuths.append(azimuth)

                
                
            bestAzimuth = azimuths[numpy.argmin(scores)]
            if robust is False:
                robustScale = 1.4826 * numpy.sqrt(numpy.median(scores))

            error = numpy.arctan2(numpy.sin((groundTruthAz-bestAzimuth)*numpy.pi/180), numpy.cos((groundTruthAz-bestAzimuth)*numpy.pi/180))*180/numpy.pi
            performance = numpy.append(performance, error)
            elevations = numpy.append(elevations, elevation)
            bestAzimuths = numpy.append(bestAzimuths, bestAzimuth)
            groundTruthAzimuths = numpy.append(groundTruthAzimuths, groundTruthAz)

            cv2.imwrite(numDir + 'bestImage' + "_canny_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png' , bestImageEdges)
            cv2.imwrite(numDir + 'bestImage' + "_az" + '%.1f' % bestAzimuth + '_dist' + '%.1f' % score + '.png', numpy.uint8(bestImage*255.0))

            imgEdges = cv2.Canny(numpy.uint8(testImage*255), minThresImage,maxThresImage)
            bwEdges1 = cv2.distanceTransform(~imgEdges, cv2.DIST_L2, 5)
            disp = cv2.normalize(bwEdges1, bwEdges1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(numDir + 'dist_transform' +  '.png', disp)

            plt.plot(azimuths, numpy.array(scores))
            plt.xlabel('Azimuth (degrees)')
            plt.ylabel('Distance')
            plt.title('Chamfer distance')
            plt.axvline(x=bestAzimuth, linewidth=2, color='b', label='Minimum distance azimuth')
            plt.axvline(x=groundTruthAz, linewidth=2, color='g', label='Ground truth azimuth')
            plt.axvline(x=(bestAzimuth + 180) % 360, linewidth=1, color='b', ls='--', label='Minimum distance azimuth + 180')
            fontP = FontProperties()
            fontP.set_size('small')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,360,0,y2))
            # plt.legend()
            plt.savefig(numDir + 'performance.png')
            plt.clf()


        experiment = {'methodParams': methodParams, 'distanceType': distanceType, 'teapot':teapot, 'bestAzimuths':bestAzimuths, 'performance': performance, 'elevations':elevations, 'groundTruthAzimuths': groundTruthAzimuths, 'selTest':selTest, 'expSelTest':expSelTest}
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

        # Cleanup
        # for obji in scene.objects:
        #     if obji.type == 'MESH':
        #         obji.user_clear()
        #         bpy.data.objects.remove(obji)

        # scene.user_clear()
        # bpy.ops.scene.delete()   

print("Finished the experiment")

     


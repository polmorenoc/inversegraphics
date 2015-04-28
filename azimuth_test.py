#!/usr/bin/env python3.4m
 
from utils import *

bpy.ops.render.render( write_still=True )

baseDir = '../databaseFull/models/'

lines = [line.strip() for line in open('teapots.txt')]

for object in bpy.data.scenes['Scene'].objects: print(object.name)

lamp = bpy.data.scenes['Scene'].objects[1]
lamp.location = (0,0.0,1.5)

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
world.light_settings.environment_energy = 1
world.horizon_color = mathutils.Color((0.0,0.0,0.0))

width = 200
height = 200

data, images, experiments = loadData()

groundTruthEls = data['altitudes'][0][0][0]
groundTruthAzs = data['azimuths'][0][0][0]

filenames = [name[0] for name in data['filenames'][0][0][0][:]]
ids = [name[0] for name in data['ids'][0][0][0][:]]

labels = numpy.column_stack((numpy.cos(groundTruthAzs*numpy.pi/180), numpy.sin(groundTruthAzs*numpy.pi/180), numpy.cos(groundTruthAzs*numpy.pi/180.0), numpy.sin(groundTruthAzs*numpy.pi/180.0)))

teapotTest = 'teapots/fa1fa0818738e932924ed4f13e49b59d/Teapot N300912'

indices = [i for i, s in enumerate(ids) if teapotTest in s]

output = scipy.io.loadmat('../data/crossval6div2-hog8-alldataexperiments.mat')['output_data']

selTest = indices

numTests = len(selTest)



minThresTemplate = 10
maxThresTemplate = 100
minThresImage = 50
maxThresImage = 150
performance = numpy.array([])

for selTestNum in numpy.arange(0,numTests,int(numTests/10)):

    test = selTest[selTestNum]
    groundTruthAz = groundTruthAzs[test]
    scores = []
    azimuths = []

    rgbTestImage = numpy.transpose(images["images"][test])
    testImage = cv2.cvtColor(numpy.float32(rgbTestImage*255), cv2.COLOR_RGB2BGR)/255.0

    testImageEdges = cv2.Canny(numpy.uint8(testImage*255), minThresImage,maxThresImage)
    cv2.imwrite('aztest/' + str(int(groundTruthAz)) + "_canny_" +  '_' + str(test) + ".png" , testImageEdges)
    cv2.imwrite('aztest/' + str(int(groundTruthAz)) + "_image_" + '_' + str(test) + ".png" , numpy.uint8(testImage*255))

    score = 9999999
        
    teapot = teapotTest + '_cleaned'
    fullTeapot = baseDir + teapot

    print("Reading " + fullTeapot + '.dae')

    bpy.ops.scene.new()
    bpy.context.scene.name = teapot
    scene = bpy.context.scene

    scene.objects.link(lamp)

    scene.camera = camera

    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height
    scene.world = world

    scene.render.filepath = teapot + '.png'

    bpy.utils.collada_import(fullTeapot + '.dae')

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

    center = centerOfGeometry(scene)
    for mesh in scene.objects:
        if mesh.type == 'MESH':
            mesh.matrix_world = mathutils.Matrix.Translation(-center) * mesh.matrix_world

    #Rotate the object to the azimuth angle we define as 0.
    rot = mathutils.Matrix.Rotation(radians(90), 4, 'Z')
    rotateMatrixWorld(scene, rot)
    scene.update()

    camera.data.angle = 60 * 180 / numpy.pi

    elevation = groundTruthEls[test]
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
        
        scene.render.filepath = 'aztest/' + str(int(groundTruthAz))  +  teapot.replace("/", "blender_") + '_' +  str(test) +  "_az" + '%.1f' % azimuth + '_dist' + '%.1f' % distance + '.png'
        bpy.ops.render.render( write_still=False )
        # image = cv2.imread(scene.render.filepath, cv2.IMREAD_ANYDEPTH)

        blendImage = bpy.data.images['Render Result']

        image = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height/2,width/2,4]))[:,:,0:3]

        # Truncate intensities larger than 1.
        image[numpy.where(image > 1)] = 1

        image = cv2.cvtColor(numpy.float32(image*255), cv2.COLOR_RGB2BGR)/255.0

        distance = getChamferDistance(testImage, image, minThresImage, maxThresImage, minThresTemplate, maxThresTemplate)

        if distance < score:
            score = distance


        scores.append(distance)
        azimuths.append(azimuth)

        imageEdges = cv2.Canny(numpy.uint8(image*255.0), minThresTemplate,maxThresTemplate)
        directory = 'aztest/' + str(int(groundTruthAz))  + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite('aztest/' + str(int(groundTruthAz))  + '/' + teapot.replace("/", "") + '_' +  str(test) + "_canny_az" + '%.1f' % azimuth + '_dist' + '%.1f' % distance + '.png' , imageEdges)
        cv2.imwrite('aztest/' + str(int(groundTruthAz)) +  '/' + teapot.replace("/", "") + '_' +  str(test) + "_az" + '%.1f' % azimuth + '_dist' + '%.1f' % distance + '.png' , numpy.uint8(image*255.0))

    bestAzimuth = azimuths[numpy.argmin(scores)]

    error = numpy.arctan2(numpy.sin((groundTruthAz-bestAzimuth)*numpy.pi/180), numpy.cos((groundTruthAz-bestAzimuth)*numpy.pi/180))*180/numpy.pi;
    performance = numpy.append(performance, error)

    plt.plot(azimuths, numpy.array(scores))
    plt.axvline(x=bestAzimuth, linewidth=2, color='r')
    plt.axvline(x=groundTruthAz, linewidth=2, color='g')
    plt.savefig('aztest/' + str(int(groundTruthAz)) +  '/' + teapot.replace("/", "") + '_' + str(test) + '_performance.png')
    plt.clf()

    #     # Cleanup
    for obji in scene.objects:
        if obji.type == 'MESH':
            obji.user_clear()
            bpy.data.objects.remove(obji)

    scene.user_clear()
    bpy.ops.scene.delete()    


#!/usr/bin/env python3.4m
 
import bpy
import numpy
from PIL import Image
import mathutils
from math import radians
import h5py
import scipy.io
from score_image import getChamferDistance
import cv2
import sys
import io

def loadData():
    #data
    fdata = h5py.File('../data/data-all-flipped-cropped-512.mat','r')
    data = fdata["data"]

    images = h5py.File('../data/images-all-flipped-cropped-512-color.mat','r')
    # images = f["images"]
    # imgs = numpy.array(images)
    # N = imgs.shape[0]
    # imgs = imgs.transpose(0,2,3,1)

    # f = h5py.File('../data/all-flipped-cropped-512-crossval6div2_py-experiment.mat')
    # experiments = f["experiments_data"]

    # f = h5py.File('../data/all-flipped-cropped-512-crossval6div2_py-experiment.mat')
    experiments = scipy.io.loadmat('../data/all-flipped-cropped-512-crossval6all2-experiment.mat')

    return data, images, experiments['experiments_data']



def makeMaterial(name, diffuse, specular, alpha):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT' 
    mat.diffuse_intensity = 1.0 
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    return mat
 

def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def modelHeight(scene):
    maxZ = -999999;
    minZ = 99999;
    for model in scene.objects:
        if model.type == 'MESH':
            for v in model.data.vertices:
                if (model.matrix_world * v.co).z > maxZ:
                    maxZ = (model.matrix_world * v.co).z
                if (model.matrix_world * v.co).z < minZ:
                    minZ = (model.matrix_world * v.co).z


    return minZ, maxZ

def modelWidth(scene):
    maxY = -999999;
    minY = 99999;
    for model in scene.objects:
        if model.type == 'MESH':
            for v in model.data.vertices:
                if (model.matrix_world * v.co).y > maxY:
                    maxY = (model.matrix_world * v.co).y
                if (model.matrix_world * v.co).y < minY:
                    minY = (model.matrix_world * v.co).y


    return minY, maxY


def centerOfGeometry(scene):
    center = mathutils.Vector((0.0,0.0,0.0))
    numVertices = 0.0
    for model in scene.objects:
        if model.type == 'MESH':
            numVertices = numVertices + len(model.data.vertices)
            for v in model.data.vertices:
                center = center + (model.matrix_world * v.co)


    return center/numVertices

def setEulerRotation(scene, eulerVectorRotation):
    for model in scene.objects:
        if model.type == 'MESH':
            model.rotation_euler = eulerVectorRotation

    scene.update()

def rotateMatrixWorld(scene, rotationMat):
    for model in scene.objects:
        if model.type == 'MESH':
            model.matrix_world = rotationMat * model.matrix_world

    scene.update()    


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
world.light_settings.environment_energy = 0.3
world.horizon_color = mathutils.Color((0.0,0.0,0.0))

width = 200
height = 200


data, images, experiments = loadData()

# images["images"][i]

labels = numpy.column_stack((numpy.cos(numpy.array(data["altitudes"])*numpy.pi/180), numpy.sin(numpy.array(data["altitudes"])*numpy.pi/180), numpy.cos(numpy.array(data["azimuths"])*numpy.pi/180.0), numpy.sin(numpy.array(data["azimuths"])*numpy.pi/180.0)))
groundTruthEls = numpy.array(data["altitudes"])
groundTruthAzs = numpy.array(data["azimuths"])

expi = 3

experiment = experiments['experiments'][0][0][0][expi]

selTrain = experiment['selTrain'][0][0][0]

selTest = experiment['selTest'][0][0][0]

output = scipy.io.loadmat('../data/crossval6div2-hog8-alldataexperiments.mat')['output_data']

idx = output['idx'][0][0][expi][0]

nnpredazs = output['nnpredradazs'][0][0][expi][0]*180.0/numpy.pi
nnpredalts = output['nnpredradalts'][0][0][expi][0]*180.0/numpy.pi
# rtpredazs = output['rtpredradazs'][0][0][expi][0]*180.0/numpy.pi
# rtpredalts = output['rtpredradalts'][0][0][expi][0]*180.0/numpy.pi

predazs =nnpredazs.squeeze()
predalts=nnpredalts.squeeze()

numTest = selTest.size

bestModels= [""]*numTest
bestScores = numpy.ones(numTest)*999999
bestAzimuths = numpy.zeros(numTest)
bestElevations = numpy.zeros(numTest)

predi = 0

# selTest[[10384,10397,10408,10440,10442,10446,10458,10469,10478,10492]]:
# for selTestNum in [10384]:

for selTestNum in [10384, 10492]:

    test = selTest[selTestNum]
    rgbTestImage = numpy.transpose(images["images"][test])
    testImage = cv2.cvtColor(numpy.float32(rgbTestImage*255), cv2.COLOR_RGB2BGR)/255.0

    gtAzimuth = groundTruthAzs[test][0]
    testImageEdges = cv2.Canny(numpy.uint8(testImage*255), 50,150)
    cv2.imwrite("canny_" + str(test) + "az" + '%.1f' % gtAzimuth + ".png" , testImageEdges)
    cv2.imwrite("image_" + str(test) + "az" + '%.1f' % gtAzimuth + ".png" , numpy.uint8(testImage*255))

    score = 9999999

    for teapot in lines[0:3]:
        

        fullTeapot = baseDir + teapot

        print("Reading " + fullTeapot + '.dae')

        bpy.ops.scene.new()
        bpy.context.scene.name = teapot
        scene = bpy.context.scene

        scene.objects.link(lamp)

        scene.camera = camera

        # scene.render.use_raytrace = True
        # scene.render.antialiasing_samples = '16'
        

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

        camera.data.angle = 60 * 180 / numpy.pi

        stopSearchEl  = False
        stopSearchAz  = False
        dirEl = 1
        dirAz = 1

        elevation = predalts[selTestNum]
        azimuth = predazs[selTestNum]
  


        for i in numpy.arange(0,360,9):
            azimuth = i
            center = centerOfGeometry(scene)
            elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
            azimuthRot = mathutils.Matrix.Rotation(radians(azimuth), 4, 'Z')
            location = azimuthRot * elevationRot * (center + originalLoc)
            camera.location = location
            scene.update()
            look_at(camera, center)
            scene.update()
            scene.render.filepath = 'teapottests/' + str(test) + '/' +  teapot + "_az" + '%.1f' % azimuth + '.png'
            bpy.ops.render.render( write_still=True )
            image = cv2.imread('teapottests/' + str(test) + '/' +  teapot + "_az" + '%.1f' % azimuth + '.png', cv2.IMREAD_ANYDEPTH)
            image = numpy.float16(image)/255.0
            distance = getChamferDistance(testImage, image)
            # distances.append(distance)
            imageEdges = cv2.Canny(numpy.uint8(image*255.0), 25,225)
            cv2.imwrite('teapottests/' + str(test) + '/' +  teapot + "_az" + '%.1f' % azimuth + '_dist' + '%.1f' % distance + '.png' , imageEdges)            
            # azs.append(i)
            # bestAzs.append(groundTruthAzs[selTest[test]])

        
        
        

        # blendImage = bpy.data.images['Render Result']
        # image = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height/2,width/2,4]))
        # image[numpy.where(image > 1)] = 1




        # if distance < score:
        #     score = distance
        #     bestModels[predi] = teapot
        #     bestScores[predi] = score
        #     bestElevations[predi] = elevation
        #     bestAzimuths[predi] = azimuth


        # while not stopSearchEl:
        #     elevation = (elevation + dirEl*2) % 90

        #     elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
        #     location = azimuthRot * elevationRot * (center + originalLoc)

        #     camera.location = location
        #     scene.update()

        #     look_at(camera, center)
            # scene.update() 
        #     bpy.ops.render.render( write_still=True )

        #     # blendImage = bpy.data.images['Render Result']

        #     # image = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height/2,width/2,4]))

        #     # # Truncate intensities larger than 1.
        #     # image[numpy.where(image > 1)] = 1

        #     image = cv2.imread(teapot + '.png', cv2.IMREAD_ANYDEPTH)

        #     image = numpy.float16(image)/255.0

        #     distance = getChamferDistance(image, testImage)

        #     if distance < score:
        #         score = distance
        #         bestModels[predi] = teapot
        #         bestScores[predi] = score
        #         bestElevations[predi] = elevation
                
        #     elif dirEl > 0:
        #         elevation = predalts[selTestNum]
        #         dirEl = -1
        #     else:
        #         stopSearchEl = True

    #     iaz = 0
    #     azimuth = 0
    #     while not stopSearchAz:
    #         azimuth = (azimuth + dirAz*5) % 360

    #         azimuthRot = mathutils.Matrix.Rotation(radians(azimuth), 4, 'Z')
    #         location = azimuthRot * elevationRot * (center + originalLoc)
    #         camera.location = location
    #         scene.update()
    #         look_at(camera, center)          
    #         scene.update()

            
    #         bpy.ops.render.render( write_still=True )
            

    #         blendImage = bpy.data.images['Render Result']

    #         # image = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height/2,width/2,4]))[:,:,0:3]

    #         # # Truncate intensities larger than 1.
    #         # image[numpy.where(image > 1)] = 1

    #         image = cv2.imread(teapot + '.png', cv2.IMREAD_ANYDEPTH)
    #         image = numpy.float32(image)/255.0

    #         distance = getChamferDistance(testImage, image)

    #         if distance < score:
    #             score = distance
    #             bestModels[predi] = teapot
    #             bestScores[predi] = score
    #             bestAzimuths[predi] = azimuth
    #             imageEdges = cv2.Canny(numpy.uint8(image*255.0), 25,225)
    #             cv2.imwrite(teapot + "_canny_" + str(test) + ".png" , imageEdges)
    #         # elif dirAz > 0:
    #         #     azimuth = predazs[selTestNum]
    #         #     dirAz = -1
    #         # else:
    #         #     stopSearchAz = True
    #         if azimuth >= 355:
    #             stopSearchAz = True


        
        
    #     # Save best image.
    #     # im = Image.fromarray(numpy.uint8(image*255))

    #     # im.save(teapot + '.png')


    #     # Cleanup

        for obji in scene.objects:
            if obji.type == 'MESH':
                obji.user_clear()
                bpy.data.objects.remove(obji)

        scene.user_clear()
        bpy.ops.scene.delete()    

    predi = predi + 1


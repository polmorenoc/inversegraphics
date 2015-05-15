#!/usr/bin/env python3.4m
 
import sceneimport
from utils import * 

numpy.random.seed(1)

inchToMeter = 0.0254

sceneFile = '../databaseFull/scenes/scene00051.txt' 
targetIndex = 9
roomName = 'room09'

instances = sceneimport.loadScene(sceneFile)

targetParentPosition = instances[targetIndex][1]

[targetScenes, targetModels] = sceneimport.loadTargetModels()
[blenderScenes, modelInstances] = sceneimport.importBlenderScenes(instances, targetIndex)

scene = sceneimport.composeScene(modelInstances, targetIndex)

bpy.context.screen.scene = scene

scene.update()

useCycles = True

setupScene(scene, useCyles)

numFrames = 100

totalAzimuths = []
totalObjAzimuths = []
totalElevations = []
totalObjectIds = []

for teapotNum in range(3,len(targetModels)):
    
    teapot = targetModels[teapotNum]
    teapot.layers[1] = True

    azimuths = numpy.mod(numpy.random.uniform(270,450, numFrames), 360) # Avoid looking outside the room
    # objAzimuths = numpy.arange(0,360, 5) # Map it to non colliding rotations.
    objAzimuths = numpy.array([])
    while len(objAzimuths) < numFrames:
        num = numpy.random.uniform(0,360, 1)
        if not(num>= 250 and num<290) and not(num>= 80 and num<110):
            objAzimuths = numpy.append(objAzimuths, num)


    elevations = numpy.random.uniform(0,90, numFrames)

    scene.objects.link(teapot)
    teapot.layers[1] = True
    teapot.matrix_world = mathutils.Matrix.Translation(targetParentPosition)

    center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)

    original_matrix_world = teapot.matrix_world.copy()

    # ipdb.set_trace()
    for frame in range(numFrames):

        azimuth = azimuths[frame]
        objAzimuth = objAzimuths[frame]
        elevation = elevations[frame]

        bpy.context.scene.frame_set(frame)
        azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
        elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
        location = center + azimuthRot * elevationRot * originalLoc
        camera.location = location

        azimuthRot = mathutils.Matrix.Rotation(radians(-objAzimuth), 4, 'Z')
        teapot.matrix_world = mathutils.Matrix.Translation(original_matrix_world.to_translation()) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world

        scene.update()
        look_at(camera, center)
        scene.update()
        teapot.keyframe_insert(data_path="rotation_euler", frame=frame, index=-1)
        camera.keyframe_insert(data_path="location", frame=frame, index=-1)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame, index=-1)

        with open('output/groundtruth.txt', "a") as groundtruth:
            print(str(azimuth) + ' ' + str(objAzimuth) + ' ' + str(elevation) + ' ' + str(teapotNum) + ' ' + str(frame) , file = groundtruth)


    scene.frame_start = 0
    scene.frame_end = numFrames - 1
    

    scene.layers[1] = True
    scene.layers[0] = False
    scene.render.layers[0].use = False
    scene.render.layers[1].use = True

    cycles.samples = 1
    scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    # scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = 'output/scene_obj' + str(teapotNum) + '_single'

    scene.update()
    bpy.ops.render.render( animation=True )


    scene.layers[1] = False
    scene.layers[0] = True
    scene.render.layers[0].use = True
    scene.render.layers[1].use = False

    cycles.samples = 1024
    scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    # scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = 'output/scene_obj' + str(teapotNum) + '_'
    scene.update()
    bpy.ops.render.render( animation=True )



    scene.objects.unlink(teapot)

    objectIds = [teapotNum]*numFrames

    with open('output/groundtruth' + str(teapotNum) + '.txt', mode='wt', encoding='utf-8') as groundtruth:
        print(str(azimuths.tolist())[1:-1], file = groundtruth)
        print(str(objAzimuths.tolist())[1:-1], file = groundtruth)
        print(str(elevations.tolist())[1:-1], file = groundtruth)
        print(str(objectIds)[1:-1], file = groundtruth)



    totalAzimuths = totalAzimuths + azimuths.tolist()
    totalObjAzimuths = totalObjAzimuths + objAzimuths.tolist()
    totalElevations = totalElevations + elevations.tolist()
    totalObjectIds = totalObjectIds + objectIds

print("Renders ended.")

with open('output/groundtruth_total.txt', mode='wt', encoding='utf-8') as groundtruth:
    print(str(totalAzimuths)[1:-1], file = groundtruth)
    print(str(totalObjAzimuths)[1:-1], file = groundtruth)
    print(str(totalElevations)[1:-1], file = groundtruth)
    print(str(totalObjectIds)[1:-1], file = groundtruth)

with open('output/scene.pickle', 'wb') as pfile:
    pickle.dump(scene, pfile)
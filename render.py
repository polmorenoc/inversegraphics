#!/usr/bin/env python3.4m
 
import sceneimport
import re
from utils import *
from collision import *

import matplotlib.pyplot as plt

numpy.random.seed(1)

inchToMeter = 0.0254
outputDir = 'output_osiris/'

width = 110
height = 110

numSamples = 1024
useCycles = True
useGPU = False
distance = 0.45
numFrames = 1000
batchSize = 10

completeScene = True

cam = bpy.data.cameras.new("MainCamera")
camera = bpy.data.objects.new("MainCamera", cam)
world = bpy.data.worlds.new("MainWorld")

[targetScenes, targetModels] = sceneimport.loadTargetModels()
renderTeapotsList = [2]

replaceableScenesFile = '../databaseFull/fields/scene_replaceables.txt'

sceneLines = [line.strip() for line in open(replaceableScenesFile)]
sceneLineNums = numpy.arange(len(sceneLines))

prefix = ''

for sceneNum in sceneLineNums:
    sceneLine = sceneLines[sceneNum]
    sceneParts = sceneLine.split(' ')

    sceneFile = sceneParts[0]

    sceneNumber = int(re.search('.+?scene([0-9]+)\.txt', sceneFile, re.IGNORECASE).groups()[0])

    sceneFileName = re.search('.+?(scene[0-9]+\.txt)', sceneFile, re.IGNORECASE).groups()[0]

    targetIndex = int(sceneParts[1])

    instances = sceneimport.loadScene('../databaseFull/scenes/' + sceneFileName)

    targetParentPosition = instances[targetIndex][2]
    targetParentIndex = instances[targetIndex][1]

    [blenderScenes, modelInstances] = sceneimport.importBlenderScenes(instances, completeScene, targetIndex)
    targetParentInstance = modelInstances[targetParentIndex]
    roomName = ''
    for model in modelInstances:
        reg = re.compile('(room[0-9]+)')
        res = reg.match(model.name)
        if res:
            roomName = res.groups()[0]

    scene = sceneimport.composeScene(modelInstances, targetIndex)

    scene.update()
    scene.render.threads = 20
    scene.render.threads_mode = 'FIXED'
    bpy.context.screen.scene = scene

    cycles = bpy.context.scene.cycles
    scene.render.tile_x = 25
    scene.render.tile_y = 25

    originalLoc = mathutils.Vector((0,-distance , 0))

    setupScene(scene, modelInstances, targetIndex,roomName, world, distance, camera, width, height, numSamples, useCycles, useGPU)

    bpy.context.user_preferences.system.prefetch_frames = batchSize
    bpy.context.user_preferences.system.memory_cache_limit = 1000

    totalAzimuths = []
    totalObjAzimuths = []
    totalElevations = []
    totalObjectIds = []

    frameStart = 0
    frameEnd = frameStart + numFrames

    for teapotNum in renderTeapotsList:

        director = outputDir

        teapot = targetModels[teapotNum]
        teapot.layers[1] = True

        azimuths = numpy.mod(numpy.random.uniform(270,450, numFrames), 360) # Avoid looking outside the room

        # azimuths = numpy.array([])
        # while len(azimuths) < numFrames:
        #     num = numpy.random.uniform(0,360, 1)
        #     numpy.arccos(mathutils.Vector((-0.6548619270324707, 0.6106656193733215, -0.4452454447746277)) * mathutils.Vector((0.0, -1.0, 0.0)))
        #     objAzimuths = numpy.append(azimuths, num)

        # objAzimuths = numpy.arange(0,360, 5) # Map it to non colliding rotations.
        objAzimuths = numpy.array([])
        while len(objAzimuths) < numFrames:
            num = numpy.random.uniform(0,360, 1)
            # if not(num>= 250 and num<290) and not(num>= 80 and num<110):
            objAzimuths = numpy.append(objAzimuths, num)

        elevations = numpy.random.uniform(0,90, numFrames)

        scene.objects.link(teapot)
        teapot.layers[1] = True
        teapot.matrix_world = mathutils.Matrix.Translation(targetParentPosition)

        center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)

        original_matrix_world = teapot.matrix_world.copy()

        # ipdb.set_trace()
        for frame in range(frameStart, frameEnd):

            azimuth = azimuths[frame - frameStart]
            objAzimuth = objAzimuths[frame - frameStart]
            elevation = elevations[frame - frameStart]

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

        numBatches = int(numFrames / batchSize)
        for batch in range(numBatches):
            with open(director + 'groundtruth.txt', "a") as groundtruth:
                for batch_i in range(batchSize):
                    print(str(azimuths[batch * batchSize + batch_i]) + ' ' + str(objAzimuths[batch * batchSize + batch_i]) + ' ' + str(elevations[batch * batchSize + batch_i]) + ' ' + str(teapotNum) + ' ' + str(batch * batchSize + batch_i + frameStart) + ' ' + str(sceneNumber) + ' ' + str(targetIndex) + ' '  + prefix, file = groundtruth)

            scene.frame_start = frameStart + batch * batchSize
            scene.frame_end = min(frameStart + batch * batchSize + batchSize - 1, frameEnd)

            scene.layers[1] = True
            scene.layers[0] = False
            scene.render.layers[0].use = False
            scene.render.layers[1].use = True

            cycles.samples = 1
            scene.render.engine = 'CYCLES'

            scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
            # scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = director + 'render' + prefix + '_obj' + str(teapotNum) + '_' + 'scene' + str(sceneNumber) + '_target' + str(targetIndex) + '_' + 'single_'
            scene.update()
            bpy.ops.render.render( animation=True )

            scene.layers[1] = False
            scene.layers[0] = True
            scene.render.layers[0].use = True
            scene.render.layers[1].use = False

            # scene.frame_start = frameStart + batch * batchSize
            # scene.frame_end = min(frameStart + batch * batchSize + batchSize - 1, frameEnd)

            scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
            #scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = director  + 'render' + prefix + '_obj' + str(teapotNum) + '_' + 'scene' + str(sceneNumber) + '_target' + str(targetIndex) + '_'
            scene.update()
            bpy.ops.render.render( animation=True )

            if useCycles:
                cycles.samples = numSamples

            if not useCycles:
                scene.render.engine = 'BLENDER_RENDER'

            scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = director + 'images/' +  'render' + prefix  + '_obj' + str(teapotNum) + '_' + 'scene' + str(sceneNumber) + '_target' + str(targetIndex) + '_'
            scene.update()
            bpy.ops.render.render( animation=True )

        scene.objects.unlink(teapot)

        objectIds = [teapotNum]*numFrames

        # with open(director +  'groundtruth' + str(teapotNum) + '.txt', mode='wt', encoding='utf-8') as groundtruth:
        #     print(str(azimuths.tolist())[1:-1], file = groundtruth)
        #     print(str(objAzimuths.tolist())[1:-1], file = groundtruth)
        #     print(str(elevations.tolist())[1:-1], file = groundtruth)
        #     print(str(objectIds)[1:-1], file = groundtruth)

        totalAzimuths = totalAzimuths + azimuths.tolist()
        totalObjAzimuths = totalObjAzimuths + objAzimuths.tolist()
        totalElevations = totalElevations + elevations.tolist()
        totalObjectIds = totalObjectIds + objectIds

    # Cleanup
    for objnum, obji in enumerate(scene.objects):
        if obji.name != teapot.name:
            obji.user_clear()
            bpy.data.objects.remove(obji)

    scene.user_clear()
    bpy.data.scenes.remove(scene)
    # bpy.ops.scene.delete()


print("Renders ended.")

# with open(director + 'groundtruth_total.txt', mode='wt', encoding='utf-8') as groundtruth:
#     print(str(totalAzimuths)[1:-1], file = groundtruth)
#     print(str(totalObjAzimuths)[1:-1], file = groundtruth)
#     print(str(totalElevations)[1:-1], file = groundtruth)
#     print(str(totalObjectIds)[1:-1], file = groundtruth)
#
# with open(director + 'scene.pickle', 'wb') as pfile:
#     pickle.dump(scene, pfile)
#!/usr/bin/env python3.4m

import sceneimport
import re
from utils import *
from collision import *

numpy.random.seed(1)

inchToMeter = 0.0254
outputDir = 'output/'

width = 110
height = 110

numSamples = 1024
useCycles = False
distance = 0.45

cam = bpy.data.cameras.new("MainCamera")
camera = bpy.data.objects.new("MainCamera", cam)
world = bpy.data.worlds.new("MainWorld")

[targetScenes, targetModels] = sceneimport.loadTargetModels()

renderTeapotsList = [2]

replaceableScenesFile = '../databaseFull/fields/scene_replaceables.txt'

sceneLineNums = [0,1,2,3]

sceneLines = [line.strip() for line in open(replaceableScenesFile)]
prefix = ''

sceneCollisions = []
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

    [blenderScenes, modelInstances] = sceneimport.importBlenderScenes(instances, targetIndex)
    targetParentInstance = modelInstances[targetParentIndex]
    roomName = ''

    for model in modelInstances:
        reg = re.compile('(room[0-9]+)')
        res = reg.match(model.name)
        if res:
            roomName = res.groups()[0]

    scene = sceneimport.composeScene(modelInstances, targetIndex)

    scene.update()

    bpy.context.screen.scene = scene

    originalLoc = mathutils.Vector((0,-distance , 0))

    setupScene(scene, modelInstances, targetIndex,roomName, world, distance, camera, width, height, numSamples, useCycles)

    targetCollisions = []

    for teapotNum in renderTeapotsList:

        director = outputDir

        teapot = targetModels[teapotNum]
        teapot.layers[1] = True

        # objAzimuths = numpy.arange(0,360, 5) # Map it to non colliding rotations.
        objAzimuths = numpy.array([])

        scene.objects.link(teapot)

        teapot.matrix_world = mathutils.Matrix.Translation(targetParentPosition)

        center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)

        original_matrix_world = teapot.matrix_world.copy()
        intersections = []
        for objAzimuth in numpy.arange(0,360,5):

            azimuthRot = mathutils.Matrix.Rotation(radians(-objAzimuth), 4, 'Z')
            teapot.matrix_world = mathutils.Matrix.Translation(original_matrix_world.to_translation()) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world

            intersect = False
            for sceneInstance in scene.objects:
                if sceneInstance.type == 'EMPTY' and sceneInstance != teapot and sceneInstance.name != roomName and sceneInstance != targetParentInstance:
                    intersect = instancesIntersect(teapot, sceneInstance)

            intersections = intersections + [[objAzimuth, intersect]]
            startInterval = True
            intervals = []
            initInterval = 0
            endInterval = 0
            for idx, intersection in enumerate(intersections):
                if not intersection[1]:
                    if startInterval:
                        initInterval = intersection[0]
                        startInterval = False
                else:
                    if not startInterval:
                        if idx >= 1 and intervals[idx-1] != initInterval:
                            endInterval = intersection[0]
                            intervals = intervals + [[initInterval, endInterval]]

                        startInterval = True

        targetCollisions = [teapot.name, intervals]
        scene.objects.unlink(teapot)

    sceneCollisions = [sceneNumber, targetCollisions]

    # Cleanup
    for objnum, obji in enumerate(scene.objects):
        if obji.name != teapot.name:
            obji.user_clear()
            bpy.data.objects.remove(obji)

    scene.user_clear()
    bpy.data.scenes.remove(scene)
    # bpy.ops.scene.delete()


with open('data/' + 'collisions.pickle', 'wb') as pfile:
    pickle.dump(sceneCollisions, pfile)

print("Collision detection ended.")


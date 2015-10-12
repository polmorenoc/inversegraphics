#!/usr/bin/env python3.4m

import sceneimport
import re
from utils import *
from collision import *

numpy.random.seed(1)

inchToMeter = 0.0254
outputDir = 'data/'

width = 110
height = 110

numSamples = 1024
useCycles = False
distance = 0.45

sceneimport.loadTargetsBlendData()

sceneCollisions = []
replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'
sceneLines = [line.strip() for line in open(replaceableScenesFile)]
teapots = [line.strip() for line in open('teapots.txt')]
renderTeapotsList = np.arange(len(teapots))

scaleZ = 0.265
scaleY = 0.18
scaleX = 0.3
cubeScale = mathutils.Matrix([[scaleX, 0,0,0], [0,scaleY,0 ,0] ,[0,0,scaleZ,0],[0,0,0,1]])

for sceneIdx in range(len(sceneLines)):
    sceneLine = sceneLines[sceneIdx]
    sceneParts = sceneLine.split(' ')
    sceneFile = sceneParts[0]
    sceneNumber = int(re.search('.+?scene([0-9]+)\.txt', sceneFile, re.IGNORECASE).groups()[0])
    sceneimport.loadTargetsBlendData()

    sceneimport.loadSceneBlendData(sceneIdx, replaceableScenesFile)
    scene = bpy.data.scenes['Main Scene']

    targetParentIndices, targetPositions = sceneimport.getSceneTargetParentPositions(sceneIdx, replaceableScenesFile)

    roomName = ''
    for model in scene.objects:
        reg = re.compile('(room[0-9]+)')
        res = reg.match(model.name)
        if res:
            roomName = res.groups()[0]

    cubeTarget = bpy.data.scenes['Scene'].objects['Cube']

    scene.link(cubeTarget)
    for targetIdx, targetParentIndex in enumerate(targetParentIndices):
        targetParentPosition = targetPositions[targetIdx]
        targetCollisions = []
        director = outputDir
        cubeTarget.layers[1] = True
        # objAzimuths = numpy.arange(0,360, 5) # Map it to non colliding rotations.
        objAzimuths = numpy.array([])
        scene.objects.link(cubeTarget)
        intersections = []
        translationMat = mathutils.Matrix.Translation(targetParentPosition)

        for objAzimuth in numpy.arange(0,360,5):

            azimuthRot = mathutils.Matrix.Rotation(radians(-objAzimuth), 4, 'Z')
            cubeTarget.matrix_world = translationMat * azimuthRot * cubeScale
            intersect = False
            for sceneInstanceIdx, sceneInstance in enumerate(scene.objects):
                if sceneInstance.type == 'EMPTY' and sceneInstance != cubeTarget and sceneInstance.name != roomName and (len(scene.objects) - sceneInstanceIdx - 2) != targetParentIndex:
                    intersect = instancesIntersect(cubeTarget, sceneInstance)

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

        targetCollisions = targetCollisions + [(targetParentIndex, intervals)]

    sceneCollisions = sceneCollisions + [(sceneNumber, targetCollisions)]

    bpy.ops.wm.read_homefile(load_ui=False)

    # Cleanup
    # for objnum, obji in enumerate(scene.objects):
    #     if obji.name != teapot.name:
    #         obji.user_clear()
    #         bpy.data.objects.remove(obji)
    #
    # scene.user_clear()
    # bpy.data.scenes.remove(scene)
    # bpy.ops.scene.delete()


with open('data/' + 'collisions.pickle', 'wb') as pfile:
    pickle.dump(sceneCollisions, pfile)

print("Collision detection ended.")


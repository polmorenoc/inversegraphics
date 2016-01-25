#!/usr/bin/env python3.4m

import scene_io_utils
import re
from blender_utils import *
from collision import *


numpy.random.seed(1)

inchToMeter = 0.0254
outputDir = 'data/'

width = 300
height = 300

numSamples = 1024
useCycles = False
distance = 0.75

scene_io_utils.loadTargetsBlendData()

sceneCollisions = {}
replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'
sceneLines = [line.strip() for line in open(replaceableScenesFile)]
teapots = [line.strip() for line in open('teapots.txt')]
renderTeapotsList = np.arange(len(teapots))

scaleZ = 0.265
scaleY = 0.18
scaleX = 0.35
cubeScale = mathutils.Matrix([[scaleX/2, 0,0,0], [0,scaleY/2,0 ,0] ,[0,0,scaleZ/2,0],[0,0,0,1]])

for sceneIdx in range(len(sceneLines))[7:8]:
    sceneLine = sceneLines[sceneIdx]
    sceneParts = sceneLine.split(' ')
    sceneFile = sceneParts[0]
    sceneNumber = int(re.search('.+?scene([0-9]+)\.txt', sceneFile, re.IGNORECASE).groups()[0])
    scene_io_utils.loadTargetsBlendData()
    bpy.ops.wm.read_factory_settings()
    scene_io_utils.loadSceneBlendData(sceneIdx, replaceableScenesFile)
    scene = bpy.data.scenes['Main Scene']

    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height
    scene.render.tile_x = height/2
    scene.render.tile_y = width/2
    scene.cycles.samples = 10
    sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)

    roomName = ''
    for model in scene.objects:
        reg = re.compile('(room[0-9]+)')
        res = reg.match(model.name)
        if res:
            roomName = res.groups()[0]

    cubeTarget = bpy.data.scenes['Scene'].objects['Cube']
    try:
        bpy.data.scenes['Main Scene'].objects.unlink(bpy.data.scenes['Scene'].objects['Cube'])
    except:
        pass

    scene.objects.link(cubeTarget)
    targetCollisions = {}
    for targetIdx in range(len(targetIndices)):
        targetParentIndex  = targetIndices[targetIdx]
        targetParentPosition = np.array(targetPositions[targetIdx])


        scene.objects.unlink(scene.objects[str(targetParentIndex)])

        director = outputDir
        cubeTarget.layers[1] = True
        # objAzimuths = numpy.arange(0,360, 5) # Map it to non colliding rotations.
        objAzimuths = numpy.array([])
        intersections = []
        translationMat = mathutils.Matrix.Translation(targetParentPosition)

        azimuthRot = mathutils.Matrix.Rotation(radians(0), 4, 'Z')
        cubeTarget.matrix_world = translationMat * azimuthRot * cubeScale * mathutils.Matrix.Translation(mathutils.Vector((0,0,1)))
        scene.render.filepath = 'scenes/' + str(sceneNumber) + '/collisionCubeExample_' + str(targetIdx) + '.png'
        placeCamera(scene.camera, -90, 25, 0.75, targetParentPosition)
        bpy.ops.render.render(write_still=True)

        collisionInterval = 10
        for objAzimuth in numpy.arange(0,360,collisionInterval):
            azimuthRot = mathutils.Matrix.Rotation(radians(-objAzimuth), 4, 'Z')
            cubeTarget.matrix_world = translationMat * azimuthRot * cubeScale * mathutils.Matrix.Translation(mathutils.Vector((0,0,1)))
            # scene.render.filepath = 'scenes/' + str(sceneNumber) + '/' + str(objAzimuth) + 'collisionCubeExample_' + str(targetIdx) + '.png'
            # bpy.ops.render.render(write_still=True)
            intersect = False

            for sceneInstanceIdx, sceneInstance in enumerate(scene.objects):
                if sceneInstance.type == 'EMPTY' and sceneInstance != cubeTarget and sceneInstance.name != str(roomInstanceNum) and sceneInstance.name != str(instances[targetParentIndex][1]):
                    intersect = instancesIntersect(mathutils.Matrix.Identity(4), [cubeTarget], sceneInstance.matrix_world, sceneInstance.dupli_group.objects)
                    if intersect:
                        print("Intersects with " + sceneInstance.name)
                        break

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
                    if idx >= 1 and intersections[idx-1][0] != initInterval:
                        endInterval = intersection[0] - collisionInterval
                        intervals = intervals + [[initInterval, endInterval]]
                    startInterval = True

        if not intersection[1]:
            endInterval = intersection[0]
            if not intersections[0][1]:
                intervals = intervals + [[initInterval, endInterval+collisionInterval]]
            else:
                intervals = intervals + [[initInterval, endInterval]]



        targetCollisions[targetParentIndex] = (targetParentPosition, intervals)

    sceneCollisions[sceneNumber] = targetCollisions
    with open('data/collisions/collisionScene' + str(sceneNumber) + '.pickle', 'wb') as pfile:
        pickle.dump(targetCollisions, pfile)


    print("Collision detection ended.")
# with open('data/collisions/collisions.pickle', 'wb') as pfile:
#     pickle.dump(sceneCollisions, pfile)
    # Cleanup

    # for scene in bpy.data.scenes:
    #     # for objnum, obji in enumerate(scene.objects):
    #     #
    #     #     obji.user_clear()
    #     #     bpy.data.objects.remove(obji)
    #         # scene = bpy.data.scenes['Main Scene']
    #     if scene.name != 'Scene':
    #         scene.user_clear()
    #         bpy.data.scenes.remove(scene)
    #         bpy.ops.scene.delete()




from utils import *



def loadScene(sceneNum):
    scenesFile = '../databaseFull/fields/scenes.txt'
    lines = [line.strip() for line in open(scenesFile)]
    sceneFile = lines[sceneNum]
    sceneLines = [line.strip() for line in open(scenesFile)]

    numModels = sceneLines[2].split()[1]
    instances = []
    for line in sceneLines:
        parts = line.split()
        if parts[0] == 'newModel':
            modelId = parts[2]
        if parts[0] == 'parentContactPosition':
            parentContactPosition = mathutils.Vector([float(parts[1]), float(parts[2]), float(parts[3])])            
        if parts[0] == 'transform': 
            transform = mathutils.Matrix([[float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])], [ float(parts[9]), float(parts[10]), float(parts[11]), [float(parts[12])], [float(parts[13]), float(parts[14]), float(parts[15]), float(parts[16])]])
            instances.append([modelId, parentContactPosition, transform])

    return instances


def composeScene(sceneNum, targetIndex):
    blenderScenes = importBlenderScenes(sceneNum, targetIndex)
    blenderTeapots = loadTargetModels()
    bpy.ops.scene.new()
    bpy.context.scene.name = 'Main Scene'
    scene = bpy.context.scene
    for model in blenderScenes:
        for mesh in model.objects:
            scene.objects.link(mesh)


    return scene


def importBlenderScenes(sceneNum, targetIndex):

    instances = loadScene(sceneNum)
    baseDir = '../COLLADA/'
    blenderScenes = []
    modelNum = 0
    for instance in instances:
        modelId = instance[0]
        transform = instance[1]
        modelPath = baseDir + modelId + '_cleaned.dae'
        
        if modelNum != targetIndex:
            bpy.ops.scene.new()
            bpy.context.scene.name = modelId
            scene = bpy.context.scene
            bpy.utils.collada_import(modelPath)
            blenderScenes.append(scene)

        for mesh in scene.objects:
            if mesh.type == 'MESH':
                mesh.matrix_world =  transform * mesh.matrix_world

        modelNum = modelNum + 1

    return blenderScenes

def loadTargetModels():

    teapots = [line.strip() for line in open('teapots.txt')]

    baseDir = '../databaseFull/models/'

    blenderTeapots = []
    modelNum = 0
    for teapot in teapots:
        fullTeapot = baseDir + teapot
        modelPath = fullTeapot
        bpy.ops.scene.new()
        bpy.context.scene.name = fullTeapot
        scene = bpy.context.scene
        bpy.utils.collada_import(modelPath)
        blenderTeapots.append(scene)

    return blenderTeapots
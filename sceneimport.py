from utils import *



def loadScene(sceneFile):
    sceneLines = [line.strip() for line in open(sceneFile)]

    numModels = sceneLines[2].split()[1]
    instances = []
    for line in sceneLines:
        parts = line.split()
        if parts[0] == 'newModel':
            modelId = parts[2]
        if parts[0] == 'parentContactPosition':
            parentContactPosition = mathutils.Vector([float(parts[1])*inchToMeter, float(parts[2])*inchToMeter, float(parts[3])*inchToMeter])            
        if parts[0] == 'parentIndex':
            parentIndex = int(parts[1])
        if parts[0] == 'transform':
            transform = mathutils.Matrix([[float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])], [ float(parts[9]), float(parts[10]), float(parts[11]), float(parts[12])], [float(parts[13]), float(parts[14]), float(parts[15]), float(parts[16])]]).transposed()
            # ipdb.set_trace()
            transform[0][3] = transform[0][3]*inchToMeter
            transform[1][3] = transform[1][3]*inchToMeter
            transform[2][3] = transform[2][3]*inchToMeter
            # ipdb.set_trace()
            
            instances.append([modelId, parentIndex, parentContactPosition, transform])

    return instances


def composeScene(modelInstances, targetIndex):

    bpy.ops.scene.new()
    bpy.context.scene.name = 'Main Scene'
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    modelNum = 0
    for instance in modelInstances:
        if modelNum != targetIndex:
            scene.objects.link(instance)
        modelNum = modelNum + 1


    return scene


def importBlenderScenes(instances, completeScene, targetIndex):

    baseDir = '../COLLADA/'
    blenderScenes = []
    modelInstances = []
    modelNum = 0
    for instance in instances:
        modelId = instance[0]

        reg = re.compile('(room[0-9]+)')
        isRoom = reg.match(modelId)

        if completeScene or isRoom:

            transform = instance[3]

            modelPath = baseDir + modelId + '_cleaned.obj'
            print('Importing ' + modelPath )

            # if modelNum != targetIndex:
            bpy.ops.scene.new()
            bpy.context.scene.name = modelId
            scene = bpy.context.scene

            # scene.unit_settings.system = 'METRIC'
            # bpy.utils.collada_import(modelPath)
            bpy.ops.import_scene.obj(filepath=modelPath)
            # ipdb.set_trace()
            sceneGroup = bpy.data.groups.new(modelId)

            scene.update()

            scaleMat = mathutils.Matrix.Scale(inchToMeter, 4)
            # xrotation = mathutils.Matrix.Rotation(-90,4, 'X')

            for mesh in scene.objects:
                if mesh.type == 'MESH':
                    sceneGroup.objects.link(mesh)
                    # ipdb.set_trace()
                    # mesh_transform = mesh.matrix_world
                    # mesh.matrix_world =  transform * mesh.matrix_world
                    mesh.pass_index = 0
                    # mesh.matrix_world[0][3] = mesh.matrix_world[0][3]*inchToMeter
                    # mesh.matrix_world[1][3] = mesh.matrix_world[1][3]*inchToMeter
                    # mesh.matrix_world[2][3] = mesh.matrix_world[2][3]*inchToMeter
                    # mesh.matrix_world = scaleMat * mesh.matrix_world
                     # ipdb.set_trace()
                    # mesh.data.show_double_sided = True

            modelInstance = bpy.data.objects.new(modelId, None)
            modelInstance.dupli_type = 'GROUP'
            modelInstance.dupli_group = sceneGroup
            modelInstance.matrix_world = transform
            modelInstance.pass_index = 0
            modelInstances.append(modelInstance)
            modelNum = modelNum + 1
            # ipdb.set_trace()
            blenderScenes.append(scene)


    return blenderScenes, modelInstances

def loadTargetModels(experimentTeapots):

    teapots = [line.strip() for line in open('teapots.txt')]
    targetModels = []

    baseDir = '../databaseFull/models/'
    targetInstances = []
    blenderTeapots = []
    transformations = []
    modelNum = 0

    selection = [ teapots[i] for i in experimentTeapots]
    for teapot in selection:
        targetGroup = bpy.data.groups.new(teapot)
        fullTeapot = baseDir + teapot + '.dae'
        modelPath = fullTeapot
        bpy.ops.scene.new()
        bpy.context.scene.name = teapot
        scene = bpy.context.scene
        scene.unit_settings.system = 'METRIC'
        print("Importing " + modelPath)
        bpy.utils.collada_import(modelPath)
        scene.update()
        modifySpecular(scene, 0.3)

        #Rotate the object to the azimuth angle we define as 0.
        # rot = mathutils.Matrix.Rotation(radians(-90), 4, 'X')
        # rot = mathutils.Matrix.Rotation(radians(90), 4, 'Z')
        # rotateMatrixWorld(scene,  rot )
        # rot = mathutils.Matrix.Rotation(radians(90), 4, 'Z')

        matrix_world = mathutils.Matrix.Identity(4)
        minZ, maxZ = modelHeight(scene.objects, mathutils.Matrix.Identity(4))
        minY, maxY = modelDepth(scene.objects, mathutils.Matrix.Identity(4))
        scaleZ = 0.265/(maxZ-minZ)
        scaleY = 0.18/(maxY-minY)
        scale = min(scaleZ, scaleY)
        scaleMat = mathutils.Matrix.Scale(scale, 4)
        for mesh in scene.objects:
            if mesh.type == 'MESH':

                mesh.matrix_world =  scaleMat * mesh.matrix_world

        matrix_world = scaleMat * matrix_world

        rot = mathutils.Matrix.Rotation(radians(90), 4, 'Z') 
        rotateMatrixWorld(scene,  rot )

        matrix_world  = rot * matrix_world

        minZ, maxZ = modelHeight(scene.objects, mathutils.Matrix.Identity(4))

        center = centerOfGeometry(scene.objects, mathutils.Matrix.Identity(4))

        for mesh in scene.objects:
            if mesh.type == 'MESH':
                mesh.matrix_world = mathutils.Matrix.Translation(-center) * mesh.matrix_world

        matrix_world = mathutils.Matrix.Translation(-center) * matrix_world

        minZ, maxZ = modelHeight(scene.objects, mathutils.Matrix.Identity(4))

        for mesh in scene.objects:
            if mesh.type == 'MESH':
                mesh.matrix_world = mathutils.Matrix.Translation(mathutils.Vector((0,0,-minZ))) * mesh.matrix_world

        matrix_world = mathutils.Matrix.Translation(mathutils.Vector((0,0,-minZ))) * matrix_world

        transformations = transformations + [matrix_world]
        for mesh in scene.objects:
            targetGroup.objects.link(mesh)
            mesh.pass_index = 1

        targetInstance = bpy.data.objects.new(teapot, None)
        targetInstance.dupli_type = 'GROUP'
        targetInstance.dupli_group = targetGroup
        targetInstance.pass_index = 1
        targetInstances.append(targetInstance)
        blenderTeapots.append(scene)
    # ipdb.set_trace()
    return blenderTeapots, targetInstances, transformations





        # ipdb.set_trace()

        

        



        


                     
        


        

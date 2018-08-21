from blender_utils import *
from sklearn.preprocessing import normalize
from collections import OrderedDict


def loadTeapotsOpenDRData(renderTeapotsList, useBlender, unpackModelsFromBlender, targetModels):
    v_teapots = []
    f_list_teapots = []
    vc_teapots = []
    vn_teapots = []
    uv_teapots = []
    haveTextures_list_teapots = []
    textures_list_teapots = []
    blender_teapots = []
    center_teapots = []
    for teapotIdx, teapotNum in enumerate(renderTeapotsList):

        objectDicFile = 'data/target' + str(teapotNum) + '.pickle'
        if useBlender:
            teapot = targetModels[teapotIdx]
            teapot.layers[1] = True
            teapot.layers[2] = True
            blender_teapots = blender_teapots + [teapot]
        if unpackModelsFromBlender:
            vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = unpackBlenderObject(teapot, objectDicFile, True)
        else:
            vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = loadSavedObject(objectDicFile)
        v_teapots = v_teapots + [vmod]
        f_list_teapots = f_list_teapots + [fmod_list]
        vc_teapots = vc_teapots + [vcmod]
        vn_teapots = vn_teapots + [vnmod]
        uv_teapots = uv_teapots + [uvmod]
        haveTextures_list_teapots = haveTextures_list_teapots + [haveTexturesmod_list]
        textures_list_teapots = textures_list_teapots + [texturesmod_list]
        vflat = [item for sublist in vmod for item in sublist]
        varray = np.vstack(vflat)
        center_teapots = center_teapots + [np.sum(varray, axis=0)/len(varray)]

    return v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots

def loadMugsOpenDRData(mugFiles, useBlender, unpackModelsFromBlender, mugModels=None):
    v_mugs = []
    f_list_mugs = []
    vc_mugs = []
    vn_mugs = []
    uv_mugs = []
    haveTextures_list_mugs = []
    textures_list_mugs = []
    blender_mugs = []
    center_mugs = []
    for mugIdx, mugNum in enumerate(mugFiles):

        objectDicFile = 'data/mug' + str(mugNum) + '.pickle'
        if useBlender:
            mug = mugModels[mugIdx]
            mug.layers[1] = True
            mug.layers[2] = True
            blender_mugs = blender_mugs + [mug]
        if unpackModelsFromBlender:
            vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = unpackBlenderObject(mug, objectDicFile, True)
        else:
            vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = loadSavedObject(objectDicFile)
        v_mugs = v_mugs + [vmod]
        f_list_mugs = f_list_mugs + [fmod_list]
        vc_mugs = vc_mugs + [vcmod]
        vn_mugs = vn_mugs + [vnmod]
        uv_mugs = uv_mugs + [uvmod]
        haveTextures_list_mugs = haveTextures_list_mugs + [haveTexturesmod_list]
        textures_list_mugs = textures_list_mugs + [texturesmod_list]
        vflat = [item for sublist in vmod for item in sublist]
        varray = np.vstack(vflat)
        center_mugs = center_mugs + [np.sum(varray, axis=0)/len(varray)]

    return v_mugs, f_list_mugs, vc_mugs, vn_mugs, uv_mugs, haveTextures_list_mugs, textures_list_mugs, vflat, varray, center_mugs

def getSceneInstancesInfo(sceneFile):
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


def composeScene(modelInstances):
    bpy.context.scene.name = 'Main Scene'
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    modelNum = 0
    for instanceidx, instance in enumerate(modelInstances):
        # if modelNum != targetIndex:
        instance.name = str(instanceidx)
        scene.objects.link(instance)
        # modelNum = modelNum + 1
    return scene

def importBlenderScenes(instances, completeScene):
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
            bpy.ops.import_scene.obj(filepath=modelPath, split_mode='OFF', use_split_objects=True, use_split_groups=False)
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
                    mesh.data.update()
            modelInstance = bpy.data.objects.new(modelId, None)
            modelInstance.dupli_type = 'GROUP'
            modelInstance.dupli_group = sceneGroup
            modelInstance.matrix_world = transform
            modelInstance.pass_index = 0
            modelInstances.append(modelInstance)
            modelNum = modelNum + 1
            # ipdb.set_trace()
            scene.update()
            blenderScenes.append(scene)

    return blenderScenes, modelInstances

import os.path

def createTargetsBlendFile():
    bpy.ops.wm.read_factory_settings()
    # bpy.ops.wm.read_homefile(load_ui=False)
    teapots = [line.strip() for line in open('teapots.txt')]
    renderTeapotsList = np.arange(len(teapots))
    [targetScenes, targetModels, transformations] = loadTargetModels(renderTeapotsList)
    # bpy.data.scenes.remove(bpy.data.scenes['Scene'])
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath='data/targets.blend')
    bpy.ops.wm.read_factory_settings()

def createMugsBlendFile():
    bpy.ops.wm.read_factory_settings()
    # bpy.ops.wm.read_homefile(load_ui=False)
    mugs = [line.strip() for line in open('mugs.txt')]
    renderMugsList = np.arange(len(mugs))
    [targetScenes, targetModels, transformations] = loadMugsModels(renderMugsList)
    # bpy.data.scenes.remove(bpy.data.scenes['Scene'])
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath='data/mugs.blend')
    bpy.ops.wm.read_factory_settings()

def createSceneBlendFiles(overwrite=True):
    replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'
    sceneLines = [line.strip() for line in open(replaceableScenesFile)]
    for sceneIdx in range(len(sceneLines)):
        sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = getSceneInformation(sceneIdx, replaceableScenesFile)
        blendFilename = 'data/scene' + str(sceneNumber) + '.blend'
        if overwrite or not os.path.isfile(blendFilename):
            print("Importing Scene " + sceneFileName)
            scene = loadBlenderScene(sceneIdx, replaceableScenesFile)
            setupScene(scene, roomInstanceNum, scene.world, scene.camera, 100, 100, 16, True, False)
            scene.update()
            # bpy.data.scenes.remove(bpy.data.scenes['Scene'])
            bpy.ops.file.pack_all()
            bpy.ops.wm.save_as_mainfile(filepath=blendFilename)
            bpy.ops.wm.read_factory_settings()
        # bpy.ops.wm.read_homefile(load_ui=False)
        # bpy.ops.wm.open_mainfile(filepath='data/Scene.blend')

def createSceneOpenDRFiles(overwrite=True):
    replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'
    sceneLines = [line.strip() for line in open(replaceableScenesFile)]
    for sceneIdx in range(len(sceneLines)):
        sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = getSceneInformation(sceneIdx, replaceableScenesFile)
        pickleFilename = 'data/scene' + str(sceneNumber) + '.pickle'
        if overwrite or not os.path.isfile(pickleFilename):
            print("Unpacking Scene " + str(sceneNumber))
            loadSceneBlendData(sceneIdx, replaceableScenesFile)
            scene = bpy.data.scenes['Main Scene']
            v, f_list, vc, vn, uv, haveTextures_list, textures_list = unpackBlenderScene(scene, pickleFilename, True)
            bpy.ops.wm.read_factory_settings()


def loadSceneBlendData(sceneIdx, replaceableScenesFile):
    replaceableScenesFile = replaceableScenesFile
    sceneLines = [line.strip() for line in open(replaceableScenesFile)]
    sceneLine = sceneLines[sceneIdx]
    sceneParts = sceneLine.split(' ')
    sceneFile = sceneParts[0]
    sceneNumber = int(re.search('.+?scene([0-9]+)\.txt', sceneFile, re.IGNORECASE).groups()[0])
    sceneFilename = 'data/scene' + str(sceneNumber) + '.blend'
    with bpy.data.libraries.load(filepath=sceneFilename) as (data_from, data_to):
        for attr in dir(data_to):
            setattr(data_to, attr, getattr(data_from, attr))

    bpy.context.screen.scene = bpy.data.scenes['Main Scene']

def loadTargetsBlendData():
    targetsFilename = 'data/targets.blend'
    with bpy.data.libraries.load(filepath=targetsFilename) as (data_from, data_to):
        for attr in dir(data_to):
            setattr(data_to, attr, getattr(data_from, attr))

        return data_to

def loadMugsBlendData():
    targetsFilename = 'data/mugs.blend'
    with bpy.data.libraries.load(filepath=targetsFilename) as (data_from, data_to):
        for attr in dir(data_to):
            setattr(data_to, attr, getattr(data_from, attr))

        return data_to

def loadTargetModels(experimentTeapots):

    teapots = [line.strip() for line in open('teapots.txt')]
    targetModels = []

    baseDir = '../databaseFull/models/'
    targetInstances = []
    blenderTeapots = []
    transformations = []
    modelNum = 0

    selection = [ teapots[i] for i in experimentTeapots]
    for teapotidx, teapot in enumerate(selection):
        targetGroup = bpy.data.groups.new(teapot)
        fullTeapot = baseDir + teapot + '.obj'
        modelPath = fullTeapot
        bpy.ops.scene.new()
        bpy.context.scene.name = teapot
        scene = bpy.context.scene
        scene.unit_settings.system = 'METRIC'
        print("Importing " + modelPath)
        # bpy.utils.collada_import(modelPath)

        bpy.ops.import_scene.obj(filepath=modelPath, split_mode='OFF', use_split_objects=True, use_split_groups=False)
        scene.update()
        # modifySpecular(scene, 0.3)

        matrix_world = mathutils.Matrix.Identity(4)
        minZ, maxZ = modelHeight(scene.objects, mathutils.Matrix.Identity(4))
        minY, maxY = modelDepth(scene.objects, mathutils.Matrix.Identity(4))

        scaleZ = 0.265/(maxZ-minZ)
        scaleY = 0.18/(maxY-minY)

        ratio =  (maxZ-minZ)/(maxY-minY)


        if ratio > 0.265/0.18:
            scale = scaleZ
        else:
            scale = scaleY
        # scale = min(scaleZ, scaleY)

        scaleMat = mathutils.Matrix.Scale(scale, 4)
        for mesh in scene.objects:
            if mesh.type == 'MESH':

                mesh.matrix_world =  scaleMat * mesh.matrix_world
                mesh.data.update()

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
            # mesh.update()
            targetGroup.objects.link(mesh)
            mesh.pass_index = 1

        targetInstance = bpy.data.objects.new(teapot, None)
        targetInstance.dupli_type = 'GROUP'
        targetInstance.dupli_group = targetGroup
        targetInstance.pass_index = 1
        targetInstances.append(targetInstance)
        targetInstance.name = 'teapotInstance' + str(experimentTeapots[teapotidx])
        scene.objects.link(targetInstance)
        scene.update()
        blenderTeapots.append(scene)
    # ipdb.set_trace()
    return blenderTeapots, targetInstances, transformations


def loadMugsModels(experimentMugs):

    mugs = [line.strip() for line in open('mugs.txt')]
    targetModels = []

    baseDir = '../databaseFull/models/'
    targetInstances = []
    blenderMugs = []
    transformations = []
    modelNum = 0

    selection = [ mugs[i] for i in experimentMugs]
    for mugidx, mug in enumerate(selection):
        targetGroup = bpy.data.groups.new(mug)
        fullTeapot = baseDir + mug + '.obj'
        modelPath = fullTeapot
        bpy.ops.scene.new()
        bpy.context.scene.name = mug
        scene = bpy.context.scene
        scene.unit_settings.system = 'METRIC'
        print("Importing " + modelPath)
        # bpy.utils.collada_import(modelPath)

        bpy.ops.import_scene.obj(filepath=modelPath, split_mode='OFF', use_split_objects=True, use_split_groups=False)
        scene.update()
        # modifySpecular(scene, 0.3)

        matrix_world = mathutils.Matrix.Identity(4)
        minZ, maxZ = modelHeight(scene.objects, mathutils.Matrix.Identity(4))
        minY, maxY = modelDepth(scene.objects, mathutils.Matrix.Identity(4))

        scaleZ = 0.1/(maxZ-minZ)

        ratio =  (maxZ-minZ)/(maxY-minY)

        scale = scaleZ

        # scale = min(scaleZ, scaleY)

        scaleMat = mathutils.Matrix.Scale(scale, 4)
        for mesh in scene.objects:
            if mesh.type == 'MESH':

                mesh.matrix_world =  scaleMat * mesh.matrix_world
                mesh.data.update()

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
            # mesh.update()
            targetGroup.objects.link(mesh)
            mesh.pass_index = 1

        targetInstance = bpy.data.objects.new(mug, None)
        targetInstance.dupli_type = 'GROUP'
        targetInstance.dupli_group = targetGroup
        targetInstance.pass_index = 1
        targetInstances.append(targetInstance)
        targetInstance.name = 'mugInstance' + str(experimentMugs[mugidx])
        scene.objects.link(targetInstance)
        scene.update()
        blenderMugs.append(scene)
    # ipdb.set_trace()
    return blenderMugs, targetInstances, transformations


def getSceneInformation(sceneIdx, scenesFile):
    replaceableScenesFile = scenesFile
    sceneLines = [line.strip() for line in open(replaceableScenesFile)]
    sceneLineNums = numpy.arange(len(sceneLines))
    sceneNum =  sceneLineNums[sceneIdx]
    sceneLine = sceneLines[sceneNum]
    sceneParts = sceneLine.split(' ')
    sceneFile = sceneParts[0]
    sceneNumber = int(re.search('.+?scene([0-9]+)\.txt', sceneFile, re.IGNORECASE).groups()[0])
    sceneFileName = re.search('.+?(scene[0-9]+\.txt)', sceneFile, re.IGNORECASE).groups()[0]
    instances = getSceneInstancesInfo('../databaseFull/scenes/' + sceneFileName)
    targetPositions = []
    targetIndices = []
    for targetIndex in sceneParts[1::]:
        targetIndex = int(targetIndex)
        targetParentPosition = instances[targetIndex][2]
        targetIndices = targetIndices + [targetIndex]
        targetPositions = targetPositions + [np.array(targetParentPosition)]

    roomName = ''
    roomInstanceNum = 0
    for modelIdx, model in enumerate(instances):
        reg = re.compile('(room[0-9]+)')
        res = reg.match(model[0])
        if res:
            roomName = res.groups()[0]
            roomInstanceNum = modelIdx

    return sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions

def getSceneIdx(sceneNumber, scenesFile):
    replaceableScenesFile = scenesFile
    sceneLines = [line.strip() for line in open(replaceableScenesFile)]
    sceneLineNums = numpy.arange(len(sceneLines))
    for sceneNum in sceneLineNums:
        sceneLine = sceneLines[sceneNum]
        sceneParts = sceneLine.split(' ')
        sceneFile = sceneParts[0]
        sceneNumInFile = int(re.search('.+?scene([0-9]+)\.txt', sceneFile, re.IGNORECASE).groups()[0])
        if sceneNumInFile == sceneNumber:
            return sceneNum
    return -2

def loadBlenderScene(sceneIdx, replaceableScenesFile):
    sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = getSceneInformation(sceneIdx, replaceableScenesFile)
    # targetParentPosition = instances[targetIndex][2]
    # targetParentIndex = instances[targetIndex][1]
    cam = bpy.data.cameras.new("MainCamera")
    camera = bpy.data.objects.new("MainCamera", cam)
    world = bpy.data.worlds.new("MainWorld")

    [blenderScenes, modelInstances] = importBlenderScenes(instances, True)

    # targetParentInstance = modelInstances[targetParentIndex]
    scene = composeScene(modelInstances)
    scene.world = world
    scene.camera = camera
    # roomInstance = scene.objects[roomName]
    # roomInstance.layers[2] = True
    # targetParentInstance.layers[2] = True

    return scene

def loadSavedScene(sceneDicFile, tex_srgb2lin):
    with open(sceneDicFile, 'rb') as pfile:
        sceneDic = pickle.load(pfile)
        v = sceneDic['v']
        f_list = sceneDic['f_list']
        vc = sceneDic['vc']
        uv = sceneDic['uv']
        haveTextures_list = sceneDic['haveTextures_list']
        vn = sceneDic['vn']
        textures_list = sceneDic['textures_list']

        if tex_srgb2lin:
            textures_listflat = [item for sublist in textures_list for item in sublist]
            for texture_list in textures_listflat:
                if texture_list is not None:
                    for texture in texture_list:
                        if texture is not None:
                            srgb2lin(texture)

        print("Loaded serialized scene!")

    return v, f_list, vc, vn, uv, haveTextures_list, textures_list

def unpackBlenderScene(scene, sceneDicFile, serializeScene):
    # bpy.ops.render.render( write_still=True )
    # ipdb.set_trace()
    # v,f_list, vc, vn, uv, haveTextures_list, textures_list = unpackObjects(teapot)
    v = []
    f_list = []
    vc  = []
    vn  = []
    uv  = []
    haveTextures_list  = []
    textures_list  = []
    print("Unpacking blender data for OpenDR.")
    for modelInstance in scene.objects:
        if modelInstance.dupli_group != None:
            vmod,f_listmod, vcmod, vnmod, uvmod, haveTextures_listmod, textures_listmod = unpackBlenderObject(modelInstance, '', False)
            # gray = np.dot(np.array([0.3, 0.59, 0.11]), vcmod[0].T).T
            # sat = 0.5
            # vcmod[0][:,0] = vcmod[0][:,0] * sat + (1-sat) * gray
            # vcmod[0][:,1] = vcmod[0][:,1] * sat + (1-sat) * gray
            # vcmod[0][:,2] = vcmod[0][:,2] * sat + (1-sat) * gray
            v = v + vmod
            f_list = f_list + f_listmod
            vc = vc + vcmod
            vn = vn + vnmod
            uv = uv + uvmod
            haveTextures_list = haveTextures_list + haveTextures_listmod
            textures_list = textures_list + textures_listmod

    #Serialize
    if serializeScene:
        sceneDic = {'v':v,'f_list':f_list,'vc':vc,'uv':uv,'haveTextures_list':haveTextures_list,'vn':vn,'textures_list': textures_list}
        with open(sceneDicFile, 'wb') as pfile:
            pickle.dump(sceneDic, pfile)

    print("Serialized scene!")

    return v, f_list, vc, vn, uv, haveTextures_list, textures_list

def loadSavedObject(objectDicFile):
    with open(objectDicFile, 'rb') as pfile:
        targetDic = pickle.load(pfile)
        v = targetDic['v']
        f_list = targetDic['f_list']
        vc = targetDic['vc']
        uv = targetDic['uv']
        haveTextures_list = targetDic['haveTextures_list']
        vn = targetDic['vn']
        textures_list = targetDic['textures_list']
    print("Loaded serialized target!")
    return [v], [f_list], [vc], [vn], [uv], [haveTextures_list], [textures_list]


def unpackBlenderObject(object, objectDicFile, saveData):
    f_list = []
    v = []
    vc = []
    vn = []
    uv = []
    haveTextures = []
    textures_list = []
    vertexMeshIndex = 0
    for mesh in object.dupli_group.objects:
        if mesh.type == 'MESH':
            # mesh.data.validate(verbose=True, clean_customdata=True)
            fmesh, vmesh, vcmesh,  nmesh, uvmesh, haveTexture, textures  = buildData(mesh.data)
            f_list = f_list + [fmesh]
            vc = vc + [vcmesh]
            transf = np.array(np.dot(object.matrix_world, mesh.matrix_world))
            vmesh = np.hstack([vmesh, np.ones([vmesh.shape[0],1])])
            vmesh = ( np.dot(transf , vmesh.T)).T[:,0:3]
            v = v + [vmesh]
            transInvMat = np.linalg.inv(transf).T
            nmesh = np.hstack([nmesh, np.ones([nmesh.shape[0],1])])
            nmesh = (np.dot(transInvMat , nmesh.T)).T[:,0:3]
            vn = vn + [normalize(nmesh, axis=1)]
            uv = uv + [uvmesh]
            haveTextures_list = haveTextures + [haveTexture]
            textures_list = textures_list + [textures]

            vertexMeshIndex = vertexMeshIndex + len(vmesh)
    #Serialize
    if saveData:
        targetDic = {'v':v,'f_list':f_list,'vc':vc,'uv':uv,'haveTextures_list':haveTextures_list,'vn':vn,'textures_list': textures_list}
        with open(objectDicFile, 'wb') as pfile:
            pickle.dump(targetDic, pfile)

        print("Serialized object!")

    return [v],[f_list],[vc],[vn], [uv], [haveTextures_list], [textures_list]

def buildData (msh):

    lvdic = {} # local dictionary
    lfl = [] # lcoal faces index list
    lvl = [] # local vertex list
    lvcl = []
    lnl = [] # local normal list
    luvl = [] # local uv list
    lvcnt = 0 # local vertices count
    isSmooth = False
    texdic = {} # local dictionary
    msh.calc_tessface()
    # if len(msh.tessfaces) == 0 or msh.tessfaces is None:
    #     msh.calc_tessface()
    textureNames = []
    haveUVs = []

    for i,f in enumerate(msh.polygons):
        isSmooth = f.use_smooth
        tmpfaces = []
        hasUV = False    # true by default, it will be verified below
        texture = None
        texname = None
        if (len(msh.tessface_uv_textures)>0):
            activeUV = msh.tessface_uv_textures.active.data

            if msh.tessface_uv_textures.active.data[i].image is not None:
                # ipdb.set_trace()
                texname = msh.tessface_uv_textures.active.data[i].image.name
                hasUV = True
                texture = texdic.get(texname)
                if (texture is None): # vertex not found
                    # print("Image: " + texname)
                    # print("Clamp x: " + str(msh.tessface_uv_textures.active.data[i].image.use_clamp_x))
                    # print("Clamp y: " + str(msh.tessface_uv_textures.active.data[i].image.use_clamp_y))
                    # print("Tile x: " + str(msh.tessface_uv_textures.active.data[i].image.tiles_x))
                    # print("Tile y: " + str(msh.tessface_uv_textures.active.data[i].image.tiles_y))
                    texture = np.flipud(np.array(msh.tessface_uv_textures.active.data[i].image.pixels).reshape([msh.tessface_uv_textures.active.data[i].image.size[1],msh.tessface_uv_textures.active.data[i].image.size[0],4])[:,:,:3])
                    texture = srgb2lin(texture)
                    if np.any(np.isnan(texture)) or np.any(texture<0) or np.any(texture>1) or texture.size == 0:
                        print("Problem with texture from Blender")
                        texture = np.flipud(np.array(msh.tessface_uv_textures.active.data[i].image.pixels).reshape([msh.tessface_uv_textures.active.data[i].image.size[1],msh.tessface_uv_textures.active.data[i].image.size[0],4])[:,:,:3])
                        hasUV = False
                        texture = None
                        texname = None
                    if hasUV:
                        texdic[texname] = texture
        textureNames = textureNames + [texname]
        haveUVs = haveUVs + [hasUV]

        for j,v in enumerate(f.vertices):
            vec = msh.vertices[v].co
            vec = r3d(vec)
            if (isSmooth):  # use vertex normal
                nor = msh.vertices[v].normal
            else:           # use face normal
                nor = f.normal

            vcolor = msh.materials[f.material_index].diffuse_color[:]
            if vcolor == (0.0,0.0,0.0) and msh.materials[f.material_index].specular_color[:] != (0.0,0.0,0.0):
                vcolor = msh.materials[f.material_index].specular_color[:]
                # print("Using specular!")
            nor = r3d(nor)
            co = (0.0, 0.0)
            if hasUV:
                co = activeUV[i].uv[j]
                co = r2d(co)
                vcolor = (1.0,1.0,1.0)
            key = vec, nor, co
            vinx = lvdic.get(key)
            if (vinx is None): # vertex not found

                lvdic[key] = lvcnt
                lvl.append(vec)
                lnl.append(nor)

                lvcl.append(vcolor)
                luvl.append(co)
                tmpfaces.append(lvcnt)
                lvcnt+=1
            else:
                inx = lvdic[key]
                tmpfaces.append(inx)

        if (len(tmpfaces)==3):
            lfl.append(tmpfaces)
        else:
            lfl.append([tmpfaces[0], tmpfaces[1], tmpfaces[2]])
            lfl.append([tmpfaces[0], tmpfaces[2], tmpfaces[3]])

    # vtx.append(lvdic)
    textures = []
    haveTextures = []
    f_list = []

    orderedtexs = OrderedDict(sorted(texdic.items(), key=lambda t: t[0]))
    for texname, texture in orderedtexs.items():
        fidxs = [lfl[idx] for idx in range(len(lfl)) if textureNames[idx] == texname]
        f_list = f_list + [np.vstack(fidxs)]
        textures = textures + [texture]
        haveTextures = haveTextures + [True]
    try:
        fidxs = [lfl[idx] for idx in range(len(lfl)) if haveUVs[idx] == False]
    except:
        ipdb.set_trace()

    if fidxs != None and fidxs != []:
        f_list = f_list + [np.vstack(fidxs)]
        textures = textures + [None]
        haveTextures = haveTextures + [False]

    #update global lists and dictionaries
    v = np.vstack(lvl)
    vc = np.vstack(lvcl)
    n = np.vstack(lnl)
    uv = np.vstack(luvl)

    return f_list, v, vc, n, uv, haveTextures, textures

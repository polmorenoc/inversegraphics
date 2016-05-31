__author__ = 'pol'

from utils import *
import opendr
import chumpy as ch
import geometry
import bpy
import mathutils
import numpy as np
from math import radians
from opendr.camera import ProjectPoints
from opendr.renderer import TexturedRenderer
from opendr.lighting import SphericalHarmonics
from opendr.lighting import LambertianPointLight
import ipdb
import light_probes
import scene_io_utils
from blender_utils import *
import imageio
import chumpy as ch
from chumpy import depends_on, Ch
import scipy.sparse as sp

class TheanoFunOnOpenDR(Ch):
    terms = 'opendr_input_gt'
    dterms = 'opendr_input'

    initialized = False

    def compileFunctions(self, theano_output, theano_input, dim_output, theano_input_gt, theano_output_gt):
        import theano
        import theano.tensor as T
        self.prediction_fn = theano.function([theano_input], theano_output)

        # self.J, updates = theano.scan(lambda i, y,x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])
        # self.J, updates = theano.scan(lambda i, y,x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])

        self.prediction_fn_gt = theano.function([theano_input_gt], theano_output_gt)

        x = theano_input
        gt_output = T.vector('gt_output')

        # self.error = T.sum(theano_output + gt_output)
        self.error = T.sum(T.pow(theano_output.ravel() - theano_output_gt.ravel(),2))

        self.errorGrad = T.grad(self.error, x)

        from theano.compile.nanguardmode import NanGuardMode
        self.errorGrad_fun = theano.function([theano_input, theano_input_gt], self.errorGrad)
        # self.error_fun = theano.function([theano_input, theano_input_gt], self.error, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.error_fun = theano.function([theano_input, theano_input_gt], self.error)
        # self.grad = theano.function([x], self.J, updates=updates, mode='FAST_RUN')

        self.initialized = True

    def compute_r(self):
        if not self.initialized:
            self.compileFunctions()

        n_channels = 1
        if len(self.opendr_input.r.shape) == 3:
            n_channels = self.opendr_input.r.shape[2]

        h = self.opendr_input.r.shape[1]
        w = self.opendr_input.r.shape[0]
        x = self.opendr_input.r.reshape([1,n_channels,h,w]).astype(np.float32)
        x_gt = self.opendr_input_gt.reshape([1,n_channels,h,w]).astype(np.float32)
        # out_gt = self.predict_input_gt().ravel().astype(np.float32)

        output = np.array(self.error_fun(x, x_gt))

        return output.ravel()

    def predict_input(self):
        n_channels = 1
        if len(self.opendr_input.r.shape) == 3:
            n_channels = self.opendr_input.r.shape[2]

        h = self.opendr_input.r.shape[1]
        w = self.opendr_input.r.shape[0]
        x = self.opendr_input.r.reshape([1,n_channels,h,w]).astype(np.float32)
        output = self.prediction_fn(x)
        return output.ravel()

    def predict_input_gt(self):
        n_channels = 1
        if len(self.opendr_input.r.shape) == 3:
            n_channels = self.opendr_input.r.shape[2]

        h = self.opendr_input.r.shape[1]
        w = self.opendr_input.r.shape[0]
        x_gt = self.opendr_input_gt.reshape([1, n_channels, h, w]).astype(np.float32)
        output = self.prediction_fn_gt(x_gt)
        return output.ravel()

    def compute_dr_wrt(self,wrt):
        if self.opendr_input is wrt:
            if not self.initialized:
                self.compileFunctions()
            n_channels = 1
            if len(self.opendr_input.r.shape) == 3:
                n_channels = self.opendr_input.r.shape[2]

            h = self.opendr_input.r.shape[1]
            w = self.opendr_input.r.shape[0]
            x = self.opendr_input.r.reshape([1, n_channels, h, w]).astype(np.float32)
            x_gt = self.opendr_input_gt.reshape([1, n_channels, h, w]).astype(np.float32)
            jac = np.array(self.errorGrad_fun(x,x_gt)).squeeze().reshape([1,self.opendr_input.r.size])

            return sp.csr.csr_matrix(jac)
        return None

    def old_grads(self):
        import theano
        import theano.tensor as T
        self.grad_fns = [theano.function([self.theano_input], theano.gradient.grad(self.theano_output.flatten()[grad_i], self.theano_input)) for grad_i in range(self.dim_output)]
        x = self.opendr_input.r
        jac = [sp.lil_matrix(np.array(grad_fun(x[None,None, :,:].astype(np.float32))).ravel()) for grad_fun in self.grad_fns]

        return sp.vstack(jac).tocsr()

class TheanoFunFiniteDiff(Ch):
    terms = 'opendr_input_gt'
    dterms = 'opendr_input'

    initialized = False

    def compileFunctions(self, theano_output, theano_input, dim_output, theano_input_gt, theano_output_gt):
        import theano
        import theano.tensor as T
        self.prediction_fn = theano.function([theano_input], theano_output)

        # self.J, updates = theano.scan(lambda i, y,x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])
        # self.J, updates = theano.scan(lambda i, y,x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])

        self.prediction_fn_gt = theano.function([theano_input_gt], theano_output_gt)

        x = theano_input
        gt_output = T.vector('gt_output')

        # self.error = T.sum(theano_output + gt_output)
        self.error = T.sum(T.pow(theano_output.ravel() - theano_output_gt.ravel(),2))

        self.errorGrad = T.grad(self.error, x)

        from theano.compile.nanguardmode import NanGuardMode
        self.errorGrad_fun = theano.function([theano_input, theano_input_gt], self.errorGrad)
        # self.error_fun = theano.function([theano_input, theano_input_gt], self.error, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.error_fun = theano.function([theano_input, theano_input_gt], self.error)
        # self.grad = theano.function([x], self.J, updates=updates, mode='FAST_RUN')

        self.initialized = True

    def compute_r(self):
        if not self.initialized:
            self.compileFunctions()

        x = self.opendr_input.r[None,None,:,:].astype(np.float32)
        x_gt = self.opendr_input_gt.r[None,None,:,:].astype(np.float32)
        # out_gt = self.predict_input_gt().ravel().astype(np.float32)

        output = np.array(self.error_fun(x, x_gt))

        return output.ravel()

    def predict_input(self):
        x = self.opendr_input.r[None,None,:,:].astype(np.float32)
        output = self.prediction_fn(x)
        return output.ravel()

    def predict_input_gt(self):
        x = self.opendr_input_gt.r[None,None,:,:].astype(np.float32)
        output = self.prediction_fn_gt(x)
        return output.ravel()

    def compute_dr_wrt(self,wrt):
        if self.opendr_input is wrt:
            if not self.initialized:
                self.compileFunctions()

            delta = 0.01

            x = self.opendr_input.r[None,None,:,:].astype(np.float32)
            x_gt = self.opendr_input_gt.r[None,None,:,:].astype(np.float32)
            jac = np.array(self.errorGrad_fun(x,x_gt)).squeeze().reshape([1,self.opendr_input.r.size])

            #Finite differences:
            approxjacs = []
            for idx, freevar in enumerate(vars):
                f0 = self.error_fun(x, x_gt)
                oldvar = freevar.r[:].copy()
                freevar[:] = freevar.r[:] + delta
                f1 = self.error_fun(x, x_gt)
                diff = (f1 - f0) / np.abs(delta)
                freevar[:] = oldvar.copy()

                approxjacs = approxjacs + [diff]
                approxjacs = np.concatenate(approxjacs)


            ipdb.set_trace()

            return sp.csr.csr_matrix(jac)
        return None

    def old_grads(self):
        import theano
        import theano.tensor as T
        self.grad_fns = [theano.function([self.theano_input], theano.gradient.grad(self.theano_output.flatten()[grad_i], self.theano_input)) for grad_i in range(self.dim_output)]
        x = self.opendr_input.r
        jac = [sp.lil_matrix(np.array(grad_fun(x[None,None, :,:].astype(np.float32))).ravel()) for grad_fun in self.grad_fns]

        return sp.vstack(jac).tocsr()


def recoverAmbientIntensities(hdritems, gtDataset, clampedCosCoeffs):
    hdrscoeffs = np.zeros([100, 9,3])
    for hdrFile, hdrValues in hdritems:
        hdridx = hdrValues[0]
        hdrscoeffs[hdridx] = hdrValues[1]

    trainEnvMapCoeffs = hdrscoeffs[gtDataset['trainEnvMaps']]


    trainTotaloffsets = gtDataset['trainEnvMapPhiOffsets'] + gtDataset['trainObjAzsGT']

    rotations = np.vstack([light_probes.sphericalHarmonicsZRotation(trainTotaloffset)[None,:] for trainTotaloffset in trainTotaloffsets])
    rotationsRel = np.vstack([light_probes.sphericalHarmonicsZRotation(trainEnvMapPhiOffsets)[None,:] for trainEnvMapPhiOffsets in gtDataset['trainEnvMapPhiOffsets']])

    envMapCoeffsRotated = np.vstack([np.dot(rotations[i], trainEnvMapCoeffs[i,[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]][None,:] for i in range(len(rotations))])
    envMapCoeffsRotatedRel = np.vstack([np.dot(rotationsRel[i], trainEnvMapCoeffs[i,[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]][None,:] for i in range(len(rotations))])

    trainAmbientIntensityGT = gtDataset['trainComponentsGT'][:,0]/((0.3*envMapCoeffsRotated[:,0,0] + 0.59*envMapCoeffsRotated[:,0,1] + 0.11*envMapCoeffsRotated[:,0,2])*clampedCosCoeffs[0])
    trainAmbientIntensityGT1 = gtDataset['trainComponentsGT'][:,1]/((0.3*envMapCoeffsRotated[:,1,0] + 0.59*envMapCoeffsRotated[:,1,1] + 0.11*envMapCoeffsRotated[:,1,2])*clampedCosCoeffs[1])
    if np.any(trainAmbientIntensityGT != trainAmbientIntensityGT1):
        print("Problem with recovery of intensities")
    return trainAmbientIntensityGT



def SHSpherePlot():
    #Visualize plots
    ignoreEnvMaps = np.loadtxt('data/bad_envmaps.txt')
    envMapDic = {}
    SHFilename = 'data/LightSHCoefficients.pickle'

    with open(SHFilename, 'rb') as pfile:
        envMapDic = pickle.load(pfile)

    hdritems = list(envMapDic.items())[0:10]

    pEnvMapsList = []
    envMapsList = []
    width = 600
    height = 300
    for hdrFile, hdrValues in hdritems:

        hdridx = hdrValues[0]
        if hdridx not in ignoreEnvMaps:
            if not os.path.exists('light_probes/envMap' + str(hdridx)):
                os.makedirs('light_probes/envMap' + str(hdridx))

            envMapCoeffs = hdrValues[1]
            envMap = np.array(imageio.imread(hdrFile))[:,:,0:3]
            # normalize = envMap.mean()*envMap.shape[0]*envMap.shape[1]
            # envMap = 0.3*envMap[:,:,0] + 0.59*envMap[:,:,1] + 0.11*envMap[:,:,2]
            # if envMap.shape[0] != height or envMap.shape[1] != width:
            envMap = cv2.resize(src=envMap, dsize=(width,height))

            print("Processing hdridx" + str(hdridx))
            # envMapCoeffs = 0.3*envMapCoeffs[:,0] + 0.59*envMapCoeffs[:,1] + 0.11*envMapCoeffs[:,2]
            # envMapCoeffs *= normalize
            # pEnvMap = SHProjection(envMap, envMapCoeffs)
            envMapMean = envMap.mean()

            envMapCoeffs2 = light_probes.getEnvironmentMapCoefficients(envMap, 1, 0, 'equirectangular')
            # envMapCoeffs2 = 0.3*envMapCoeffs2[:,0] + 0.59*envMapCoeffs2[:,1] + 0.11*envMapCoeffs2[:,2]
            pEnvMap = SHProjection(envMap, envMapCoeffs2)

            tm = cv2.createTonemapDrago(gamma=2.2)
            tmEnvMap = tm.process(envMap)
            cv2.imwrite('light_probes/envMap' + str(hdridx) + '/texture.jpeg' , 255*tmEnvMap[:,:,[2,1,0]])
            approxProjections = []
            for coeffApprox in np.arange(9) + 1:
                approxProjection = np.sum(pEnvMap[:,:,:,0:coeffApprox], axis=3)
                approxProjectionPos =  approxProjection.copy()
                approxProjectionPos[approxProjection<0] = 0
                # tm = cv2.createTonemapDrago(gamma=2.2)
                # tmApproxProjection = tm.process(approxProjection)
                approxProjections = approxProjections + [approxProjection[None, :,:,:,None]]
                cv2.imwrite('light_probes/envMap' + str(hdridx) + '/approx' + str(coeffApprox-1) + '.jpeg', 255 * approxProjectionPos[:,:,[2,1,0]])
            approxProjections = np.concatenate(approxProjections, axis=4)
            pEnvMapsList = pEnvMapsList + [approxProjections]

            envMapsList = envMapsList + [envMap[None,:,:,:]]

    pEnvMaps = np.concatenate(pEnvMapsList, axis=0)
    envMaps = np.concatenate(envMapsList, axis=0)

    #Total sum of squares
    envMapsMean = np.mean(envMaps, axis=0)
    pEnvMapsMean = np.mean(pEnvMaps, axis=0)

    tsq = np.sum((envMaps - envMapsMean)**2)/(envMapsMean.shape[0]*envMapsMean.shape[1])

    ess = np.sum((pEnvMaps - envMapsMean[None,:,:,:,None])**2, axis=(0,1,2,3))/(envMapsMean.shape[0]*envMapsMean.shape[1])

    uess = np.sum((pEnvMaps - envMaps[:,:,:,:,None])**2, axis=(0,1,2,3))/(envMapsMean.shape[0]*envMapsMean.shape[1])

    explainedVar = ess/tsq

    tsq2 = uess + ess

    ipdb.set_trace()

    #Do something with standardized residuals.
    stdres = (pEnvMaps - pEnvMapsMean)/np.sqrt(np.sum((pEnvMaps - pEnvMapsMean)**2,axis=0))

    return explainedVar


def exportEnvMapSHImages(shCoeffsRGB, useBlender, scene, width, height, rendererGT):
    import glob
    for hdridx, hdrFile in enumerate(glob.glob("data/hdr/dataset/*")):

        envMapFilename = hdrFile
        envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
        phiOffset = 0

        phiOffsets = [0, np.pi/2, np.pi, 3*np.pi/2]
        # phiOffsets = [0, np.pi/2, np.pi, 3*np.pi/2]

        if not os.path.exists('light_probes/envMap' + str(hdridx)):
            os.makedirs('light_probes/envMap' + str(hdridx))

        cv2.imwrite('light_probes/envMap' + str(hdridx) + '/texture.png' , 255*envMapTexture[:,:,[2,1,0]])

        for phiOffset in phiOffsets:
            envMapMean = envMapTexture.mean()
            envMapCoeffs = light_probes.getEnvironmentMapCoefficients(envMapTexture, envMapMean, -phiOffset, 'equirectangular')
            shCoeffsRGB[:] = envMapCoeffs
            if useBlender:
                updateEnviornmentMap(envMapFilename, scene)
                setEnviornmentMapStrength(0.3/envMapMean, scene)
                rotateEnviornmentMap(phiOffset, scene)

            scene.render.filepath = 'light_probes/envMap' + str(hdridx) + '/blender_' + str(np.int(180*phiOffset/np.pi)) + '.png'
            bpy.ops.render.render(write_still=True)
            cv2.imwrite('light_probes/envMap' + str(hdridx) + '/opendr_' + str(np.int(180*phiOffset/np.pi)) + '.png' , 255*rendererGT.r[:,:,[2,1,0]])


def exportEnvMapSHLightCoefficients():

    import glob
    envMapDic = {}
    hdrs = glob.glob("data/hdr/dataset/*")
    hdrstorender = hdrs
    sphericalMap = False
    for hdridx, hdrFile in enumerate(hdrstorender):
        print("Processing env map" + str(hdridx))
        envMapFilename = hdrFile
        envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
        phiOffset = 0

        if not os.path.exists('light_probes/envMap' + str(hdridx)):
            os.makedirs('light_probes/envMap' + str(hdridx))

        # cv2.imwrite('light_probes/envMap' + str(hdridx) + '/texture.png' , 255*envMapTexture[:,:,[2,1,0]])

        if sphericalMap:
            envMapTexture, envMapMean = light_probes.processSphericalEnvironmentMap(envMapTexture)
            envMapCoeffs = light_probes.getEnvironmentMapCoefficients(envMapTexture, 1,  0, 'spherical')
        else:
            envMapGray = 0.3*envMapTexture[:,:,0] + 0.59*envMapTexture[:,:,1] + 0.11*envMapTexture[:,:,2]
            envMapGrayMean = np.mean(envMapGray, axis=(0,1))
            envMapTexture = envMapTexture/envMapGrayMean

            # envMapTexture = 4*np.pi*envMapTexture/np.sum(envMapTexture, axis=(0,1))
            envMapCoeffs = light_probes.getEnvironmentMapCoefficients(envMapTexture, 1, 0, 'equirectangular')
            pEnvMap = SHProjection(envMapTexture, envMapCoeffs)
            approxProjection = np.sum(pEnvMap, axis=3)

        # envMapCoeffs = light_probes.getEnvironmentMapCoefficients(envMapTexture, envMapMean, phiOffset, 'equirectangular')

        envMapDic[hdrFile] = [hdridx, envMapCoeffs]

    SHFilename = 'data/LightSHCoefficients.pickle'
    with open(SHFilename, 'wb') as pfile:
        pickle.dump(envMapDic, pfile)

def transformObject(v, vn, chScale, chObjAz, chObjDisplacement, chObjRotation, targetPosition):

    scaleMat = geometry.Scale(x=chScale[0], y=chScale[1],z=chScale[2])[0:3,0:3]
    chRotAzMat = geometry.RotateZ(a=-chObjAz)[0:3,0:3]
    transformation = ch.dot(chRotAzMat, scaleMat)
    invTranspModel = ch.transpose(ch.inv(transformation))

    objDisplacementMat = computeHemisphereTransformation(chObjRotation, 0, chObjDisplacement, np.array([0,0,0]))

    newPos = objDisplacementMat[0:3, 3]

    vtransf = []
    vntransf = []
    for mesh_i, mesh in enumerate(v):
        vtransf = vtransf + [ch.dot(v[mesh_i], transformation) + newPos + targetPosition]
        vntransf = vntransf + [ch.dot(vn[mesh_i], invTranspModel)]

    return vtransf, vntransf, newPos

def createRendererTarget(glMode, hasBackground, chAz, chEl, chDist, center, v, vc, f_list, vn, light_color, chComponent, chVColors, targetPosition, chDisplacement, width,height, uv, haveTextures_list, textures_list, frustum, win ):
    renderer = TexturedRenderer()
    renderer.set(glMode=glMode)

    vflat = [item for sublist in v for item in sublist]
    rangeMeshes = range(len(vflat))

    vnflat = [item for sublist in vn for item in sublist]

    vcflat = [item for sublist in vc for item in sublist]
    # vcch = [np.ones_like(vcflat[mesh])*chVColors.reshape([1,3]) for mesh in rangeMeshes]

    vc_list = computeSphericalHarmonics(vnflat, vcflat, light_color, chComponent)

    if hasBackground:
        dataCube, facesCube = create_cube(scale=(10,10,10), st=False, rgba=np.array([1.0, 1.0, 1.0, 1.0]), dtype='float32', type='triangles')
        verticesCube = ch.Ch(dataCube[:,0:3])
        UVsCube = ch.Ch(np.zeros([verticesCube.shape[0],2]))

        facesCube = facesCube.reshape([-1,3])
        import shape_model
        normalsCube = -shape_model.chGetNormals(verticesCube, facesCube)
        haveTexturesCube = [[False]]
        texturesListCube = [[None]]
        vColorsCube = ch.Ch(dataCube[:,3:6])

        vflat = vflat + [verticesCube]
        f_list = f_list + [[[facesCube]]]
        vnflat = vnflat + [normalsCube]
        vc_list = vc_list + [vColorsCube]
        textures_list = textures_list + [texturesListCube]
        haveTextures_list = haveTextures_list + [haveTexturesCube]
        uv = uv + [UVsCube]

    if len(vflat)==1:
        vstack = vflat[0]
    else:
        vstack = ch.vstack(vflat)

    camera, modelRotation, _ = setupCamera(vstack, chAz, chEl, chDist, center + targetPosition + chDisplacement, width, height)

    setupTexturedRenderer(renderer, vstack, vflat, f_list, vc_list, vnflat,  uv, haveTextures_list, textures_list, camera, frustum, win)
    return renderer



def createRendererGT(glMode, chAz, chEl, chDist, center, v, vc, f_list, vn, light_color, chComponent, chVColors, targetPosition, chDisplacement, width,height, uv, haveTextures_list, textures_list, frustum, win ):
    renderer = TexturedRenderer()
    renderer.set(glMode=glMode)

    vflat = [item for sublist in v for item in sublist]
    rangeMeshes = range(len(vflat))
    # vch = [ch.array(vflat[mesh]) for mesh in rangeMeshes]
    vch = vflat
    if len(vch)==1:
        vstack = vch[0]
    else:
        vstack = ch.vstack(vch)

    camera, modelRotation, _ = setupCamera(vstack, chAz, chEl, chDist, center + targetPosition + chDisplacement, width, height)
    vnflat = [item for sublist in vn for item in sublist]
    # vnch = [ch.array(vnflat[mesh]) for mesh in rangeMeshes]
    # vnchnorm = [vnch[mesh]/ch.sqrt(vnch[mesh][:,0]**2 + vnch[mesh][:,1]**2 + vnch[mesh][:,2]**2).reshape([-1,1]) for mesh in rangeMeshes]
    vcflat = [item for sublist in vc for item in sublist]
    vcch = [vcflat[mesh] for mesh in rangeMeshes]
    vcch[0] = np.ones_like(vcflat[0])*chVColors.reshape([1,3])
    # vcch[0] = vcflat[0]

    vc_list = computeSphericalHarmonics(vnflat, vcch, light_color, chComponent)
    # vc_list =  computeGlobalAndPointLighting(vch, vnch, vcch, lightPosGT, chGlobalConstantGT, light_colorGT)

    setupTexturedRenderer(renderer, vstack, vch, f_list, vc_list, vnflat,  uv, haveTextures_list, textures_list, camera, frustum, win)
    return renderer


#Old method
def generateSceneImages(width, height, envMapFilename, envMapMean, phiOffset, chAzGT, chElGT, chDistGT, light_colorGT, chComponentGT, glMode):
    replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'
    sceneLines = [line.strip() for line in open(replaceableScenesFile)]
    for sceneIdx in np.arange(len(sceneLines)):
        sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
        sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'
        bpy.ops.wm.read_factory_settings()
        scene_io_utils.loadSceneBlendData(sceneIdx, replaceableScenesFile)
        scene = bpy.data.scenes['Main Scene']
        bpy.context.screen.scene = scene
        scene_io_utils.setupScene(scene, roomInstanceNum, scene.world, scene.camera, width, height, 16, True, False)
        scene.update()
        scene.render.resolution_x = width #perhaps set resolution in code
        scene.render.resolution_y = height
        scene.render.tile_x = height/2
        scene.render.tile_y = width
        scene.cycles.samples = 100
        addEnvironmentMapWorld(envMapFilename, scene)
        setEnviornmentMapStrength(0.3/envMapMean, scene)
        rotateEnviornmentMap(phiOffset, scene)

        if not os.path.exists('scenes/' + str(sceneNumber)):
            os.makedirs('scenes/' + str(sceneNumber))
        for targetIdx, targetIndex in enumerate(targetIndices):
            targetPosition = targetPositions[targetIdx]

            rendererGT.clear()
            del rendererGT

            v, f_list, vc, vn, uv, haveTextures_list, textures_list = scene_io_utils.loadSavedScene(sceneDicFile)
            # removeObjectData(targetIndex, v, f_list, vc, vn, uv, haveTextures_list, textures_list)
            # addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])
            vflat = [item for sublist in v for item in sublist]
            rangeMeshes = range(len(vflat))
            vch = [ch.array(vflat[mesh]) for mesh in rangeMeshes]
            # vch[0] = ch.dot(vch[0], scaleMatGT) + targetPosition
            if len(vch)==1:
                vstack = vch[0]
            else:
                vstack = ch.vstack(vch)
            cameraGT, modelRotationGT = setupCamera(vstack, chAzGT, chElGT, chDistGT, targetPosition, width, height)
            # cameraGT, modelRotationGT = setupCamera(vstack, chAzGT, chElGT, chDistGT, center + targetPosition, width, height)
            vnflat = [item for sublist in vn for item in sublist]
            vnch = [ch.array(vnflat[mesh]) for mesh in rangeMeshes]
            vnchnorm = [vnch[mesh]/ch.sqrt(vnch[mesh][:,0]**2 + vnch[mesh][:,1]**2 + vnch[mesh][:,2]**2).reshape([-1,1]) for mesh in rangeMeshes]
            vcflat = [item for sublist in vc for item in sublist]
            vcch = [ch.array(vcflat[mesh]) for mesh in rangeMeshes]
            vc_list = computeSphericalHarmonics(vnchnorm, vcch, light_colorGT, chComponentGT)

            rendererGT = TexturedRenderer()
            rendererGT.set(glMode=glMode)
            setupTexturedRenderer(rendererGT, vstack, vch, f_list, vc_list, vnchnorm,  uv, haveTextures_list, textures_list, cameraGT, frustum, win)
            cv2.imwrite('scenes/' + str(sceneNumber) + '/opendr_' + str(targetIndex) + '.png' , 255*rendererGT.r[:,:,[2,1,0]])

            placeCamera(scene.camera, -chAzGT[0].r*180/np.pi, chElGT[0].r*180/np.pi, chDistGT, targetPosition)
            scene.update()
            scene.render.filepath = 'scenes/' + str(sceneNumber) + '/blender_' + str(targetIndex) + '.png'
            bpy.ops.render.render(write_still=True)


def getOcclusionFraction(renderer, id=0):
    vis_occluded = np.array(renderer.indices_image==id+1).copy().astype(np.bool)
    vis_im = np.array(renderer.image_mesh_bool([id])).copy().astype(np.bool)


    return 1. - np.sum(vis_occluded)/np.sum(vis_im)

#From http://www.ppsloan.org/publications/StupidSH36.pdf
def chZonalHarmonics(a):
    zl0 = -ch.sqrt(ch.pi)*(-1.0 + ch.cos(a))
    zl1 = 0.5*ch.sqrt(3.0*ch.pi)*ch.sin(a)**2
    zl2 = -0.5*ch.sqrt(5.0*ch.pi)*ch.cos(a)*(-1.0 + ch.cos(a))*(ch.cos(a)+1.0)
    z = [zl0, zl1, zl2]
    return ch.concatenate(z)

# http://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
chSpherical_harmonics = {
    (0, 0): lambda theta, phi: ch.Ch([0.282095]),

    (1, -1): lambda theta, phi: 0.488603 * ch.sin(theta) * ch.sin(phi),
    (1, 0): lambda theta, phi: 0.488603 * ch.cos(theta),
    (1, 1): lambda theta, phi: 0.488603 * ch.sin(theta) * ch.cos(phi),

    (2, -2): lambda theta, phi: 1.092548 * ch.sin(theta) * ch.cos(phi) * ch.sin(theta) * ch.sin(phi),
    (2, -1): lambda theta, phi: 1.092548 * ch.sin(theta) * ch.sin(phi) * ch.cos(theta),
    (2, 0): lambda theta, phi: 0.315392 * (3 * ch.cos(theta)**2 - 1),
    (2, 1): lambda theta, phi: 1.092548 * ch.sin(theta) * ch.cos(phi) * ch.cos(theta),
    (2, 2): lambda theta, phi: 0.546274 * (((ch.sin(theta) * ch.cos(phi)) ** 2) - ((ch.sin(theta) * ch.sin(phi)) ** 2))
}

#From http://www.ppsloan.org/publications/StupidSH36.pdf
def chZonalToSphericalHarmonics(z, theta, phi):
    sphCoeffs = []
    for l in np.arange(len(z)):
        for m in np.arange(np.int(-(l*2+1)/2),np.int((l*2+1)/2) + 1):
            ylm_d = chSpherical_harmonics[(l,m)](theta,phi)
            sh = np.sqrt(4*np.pi/(2*l + 1))*z[l]*ylm_d
            sphCoeffs = sphCoeffs + [sh]

    #Correct order in band l=1.
    sphCoeffs[1],sphCoeffs[3] = sphCoeffs[3],sphCoeffs[1]
    chSphCoeffs = ch.concatenate(sphCoeffs)
    return chSphCoeffs

#From http://www.ppsloan.org/publications/StupidSH36.pdf
def clampedCosineCoefficients():

    constants = []
    for l in np.arange(3):
        for m in np.arange(np.int(-(l*2+1)/2),np.int((l*2+1)/2) + 1):
            normConstant = np.pi
            if l > 1 and l % 1 == 0:
                normConstant = 0
            if l == 1:
                normConstant = 2*np.pi/3
            if l > 1 and l % 2 == 0:
                normConstant = 2*np.pi*(((-1)**(l/2.-1.))/((l+2)*(l-1)))*(np.math.factorial(l)/((2**(l))*(np.math.factorial(l/2)**2)))
                # normConstant = 0.785398

            constants = constants + [normConstant]

    return np.array(constants)

def setupCamera(v, chAz, chEl, chDist, objCenter, width, height):

    chCamModelWorld = computeHemisphereTransformation(chAz, chEl, chDist, objCenter)

    chMVMat = ch.dot(chCamModelWorld, np.array(mathutils.Matrix.Rotation(radians(270), 4, 'X')))

    chInvCam = ch.inv(chMVMat)

    modelRotation = chInvCam[0:3,0:3]

    chRod = opendr.geometry.Rodrigues(rt=modelRotation).reshape(3)
    chTranslation = chInvCam[0:3,3]

    translation, rotation = (chTranslation, chRod)
    camera = ProjectPoints(v=v, rt=rotation, t=translation, f= 1.12*ch.array([width,width]), c=ch.array([width,height])/2.0, k=ch.zeros(5))
    camera.openglMat = np.array(mathutils.Matrix.Rotation(radians(180), 4, 'X'))
    return camera, modelRotation, chMVMat


def computeHemisphereTransformation(chAz, chEl, chDist, objCenter):

    chDistMat = geometry.Translate(x=ch.Ch(0), y=-chDist, z=ch.Ch(0))
    chToObjectTranslate = geometry.Translate(x=objCenter[0], y=objCenter[1], z=objCenter[2])

    chRotAzMat = geometry.RotateZ(a=chAz)
    chRotElMat = geometry.RotateX(a=-chEl)
    chCamModelWorld = ch.dot(chToObjectTranslate, ch.dot(chRotAzMat, ch.dot(chRotElMat,chDistMat)))

    return chCamModelWorld

def computeSphericalHarmonics(vn, vc, light_color, components):
    # vnflat = [item for sublist in vn for item in sublist]
    # vcflat = [item for sublist in vc for item in sublist]
    rangeMeshes = range(len(vn))
    A_list = [SphericalHarmonics(vn=vn[mesh],
                       components=components,
                       light_color=light_color) for mesh in rangeMeshes]

    vc_list = [A_list[mesh]*vc[mesh] for mesh in rangeMeshes]
    return vc_list

def computeGlobalAndPointLighting(v, vn, vc, light_pos, globalConstant, light_color):
    # Construct point light source
    rangeMeshes = range(len(vn))
    vc_list = []
    for mesh in rangeMeshes:
        l1 = LambertianPointLight(
            v=v[mesh],
            vn=vn[mesh],
            num_verts=len(v[mesh]),
            light_pos=light_pos,
            vc=vc[mesh],
            light_color=light_color)

        vcmesh = vc[mesh]*(l1 + globalConstant)
        vc_list = vc_list + [vcmesh]
    return vc_list

def setupTexturedRenderer(renderer, vstack, vch, f_list, vc_list, vnch, uv, haveTextures_list, textures_list, camera, frustum, sharedWin=None):


    f = []
    f_listflat = [item for sublist in f_list for item in sublist]
    lenMeshes = 0

    for mesh_i, mesh in enumerate(f_listflat):
        polygonLen = 0
        for polygons in mesh:
            f = f + [polygons + lenMeshes]
            polygonLen += len(polygons)
        lenMeshes += len(vch[mesh_i])
    fstack = np.vstack(f)

    # vnflat = [item for sublist in vnch for item in sublist]
    if len(vnch)==1:
        vnstack = vnch[0]
    else:
        vnstack = ch.vstack(vnch)

    # vc_listflat = [item for sublist in vc_list for item in sublist]
    if len(vc_list)==1:
        vcstack = vc_list[0]
    else:
        vcstack = ch.vstack(vc_list)

    uvflat = [item for sublist in uv for item in sublist]
    ftstack = np.vstack(uvflat)

    texturesch = []
    textures_listflat = [item for sublist in textures_list for item in sublist]
    for texture_list in textures_listflat:
        if texture_list != None:
            for texture in texture_list:
                if texture != None:
                    texturesch = texturesch + [ch.array(texture)]

    if len(texturesch) == 0:
        texture_stack = ch.Ch([])
    elif len(texturesch) == 1:
        texture_stack = texturesch[0].ravel()
    else:
        texture_stack = ch.concatenate([tex.ravel() for tex in texturesch])


    haveTextures_listflat = [item for sublist in haveTextures_list for item in sublist]

    renderer.set(camera=camera, frustum=frustum, v=vstack, f=fstack, vn=vnstack, vc=vcstack, ft=ftstack, texture_stack=texture_stack, v_list=vch, f_list=f_listflat, vc_list=vc_list, ft_list=uvflat, textures_list=textures_listflat, haveUVs_list=haveTextures_listflat, bgcolor=ch.ones(3), overdraw=True)
    renderer.msaa = True
    renderer.sharedWin = sharedWin
    # renderer.clear()
    renderer.initGL()
    renderer.initGLTexture()


def addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list):
    v.insert(0,vmod)
    f_list.insert(0,fmod_list)
    vc.insert(0,vcmod)
    vn.insert(0,vnmod)
    uv.insert(0,uvmod)
    haveTextures_list.insert(0,haveTexturesmod_list)
    textures_list.insert(0,texturesmod_list)

def addObjectDataLast(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list):
    v.insert(-1,vmod)
    f_list.insert(-1,fmod_list)
    vc.insert(-1,vcmod)
    vn.insert(-1,vnmod)
    uv.insert(-1,uvmod)
    haveTextures_list.insert(-1,haveTexturesmod_list)
    textures_list.insert(-1,texturesmod_list)

def removeObjectData(objIdx, v, f_list, vc, vn, uv, haveTextures_list, textures_list):

    del v[objIdx]
    del f_list[objIdx]
    del vc[objIdx]
    del vn[objIdx]
    del uv[objIdx]
    del haveTextures_list[objIdx]
    del textures_list[objIdx]


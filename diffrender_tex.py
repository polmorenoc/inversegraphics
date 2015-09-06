__author__ = 'pol'

import matplotlib
matplotlib.use('Qt4Agg')

import bpy
import sceneimport
import mathutils
from math import radians
import timeit
import time
import opendr
import chumpy as ch
from opendr.lighting import LambertianPointLight
from opendr.lighting import SphericalHarmonics
from opendr.filters import gaussian_pyramid
import geometry

import numpy as np
import cv2
from utils import *
import glfw
import score_image
import matplotlib.pyplot as plt
from opendr_utils import *

plt.ion()

renderTeapotsList = [2]

[targetScenes, targetModels, transformations] = sceneimport.loadTargetModels(renderTeapotsList)
teapot = targetModels[0]
teapot.layers[1] = True
teapot.layers[2] = True

width, height = (110, 110)

angle = 60 * 180 / numpy.pi
clip_start = 0.05
clip_end = 10
camDistance = 0.4
azimuth = 275
elevation = 33

cam = bpy.data.cameras.new("MainCamera")
camera = bpy.data.objects.new("MainCamera", cam)
world = bpy.data.worlds.new("MainWorld")

replaceableScenesFile = '../databaseFull/fields/scene_replaceables.txt'
sceneLines = [line.strip() for line in open(replaceableScenesFile)]
sceneLineNums = numpy.arange(len(sceneLines))
sceneNum =  sceneLineNums[0]
sceneLine = sceneLines[sceneNum]
sceneParts = sceneLine.split(' ')
sceneFile = sceneParts[0]
sceneNumber = int(re.search('.+?scene([0-9]+)\.txt', sceneFile, re.IGNORECASE).groups()[0])
sceneFileName = re.search('.+?(scene[0-9]+\.txt)', sceneFile, re.IGNORECASE).groups()[0]
targetIndex = int(sceneParts[1])
instances = sceneimport.loadScene('../databaseFull/scenes/' + sceneFileName)
targetParentPosition = instances[targetIndex][2]
targetParentIndex = instances[targetIndex][1]
teapot.layers[1] = True
teapot.layers[2] = True
teapot.matrix_world = mathutils.Matrix.Translation(targetParentPosition)
center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
original_matrix_world = teapot.matrix_world.copy()

sceneDicFile = 'sceneDic.pickle'
sceneDic = {}
if False:
    [blenderScenes, modelInstances] = sceneimport.importBlenderScenes(instances, True, targetIndex)

    targetParentInstance = modelInstances[targetParentIndex]
    targetParentInstance.layers[2] = True

    roomName = ''
    for model in modelInstances:
        reg = re.compile('(room[0-9]+)')
        res = reg.match(model.name)
        if res:
            roomName = res.groups()[0]

    scene = sceneimport.composeScene(modelInstances, targetIndex)

    roomInstance = scene.objects[roomName]
    roomInstance.layers[2] = True
    targetParentInstance.layers[2] = True

    setupScene(scene, targetIndex,roomName, world, camDistance, camera, width, height, 16, False, False)
    scene.objects.link(teapot)

    azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
    elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
    originalLoc = mathutils.Vector((0,-camDistance, 0))
    location = center + azimuthRot * elevationRot * originalLoc
    camera = scene.camera
    camera.location = location
    scene.update()
    look_at(camera, center)
    scene.update()
    scene.render.filepath = 'opendr_blender.png'
    bpy.ops.render.render( write_still=True )

    # ipdb.set_trace()
    # v,f_list, vc, vn, uv, haveTextures_list, textures_list = sceneimport.unpackObjects(teapot)
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
            vmod,f_listmod, vcmod, vnmod, uvmod, haveTextures_listmod, textures_listmod = sceneimport.unpackObjects(modelInstance)
            # gray = np.dot(np.array([0.3, 0.59, 0.11]), vcmod[0].T).T
            # sat = 0.5
            #
            # vcmod[0][:,0] = vcmod[0][:,0] * sat + (1-sat) * gray
            # vcmod[0][:,1] = vcmod[0][:,1] * sat + (1-sat) * gray
            # vcmod[0][:,2] = vcmod[0][:,2] * sat + (1-sat) * gray
            v = v + vmod
            f_list = f_list + f_listmod
            vc = vc + vcmod
            vn = vn + vnmod
            uv = uv + uvmod
            haveTextures_list  = haveTextures_list + haveTextures_listmod
            textures_list  = textures_list + textures_listmod

    #Serialize
    sceneDic = {'v':v,'f_list':f_list,'vc':vc,'uv':uv,'haveTextures_list':haveTextures_list,'vn':vn,'textures_list': textures_list}
    with open(sceneDicFile, 'wb') as pfile:
        pickle.dump(sceneDic, pfile)

    print("Serialized scene!")
else:
    with open(sceneDicFile, 'rb') as pfile:
        sceneDic = pickle.load(pfile)
        v = sceneDic['v']
        f_list = sceneDic['f_list']
        vc = sceneDic['vc']
        uv = sceneDic['uv']
        haveTextures_list = sceneDic['haveTextures_list']
        vn = sceneDic['vn']
        textures_list = sceneDic['textures_list']

    print("Loaded serialized scene!")

vmod,fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = sceneimport.unpackObjects(teapot)

rangeMeshes = range(len(v))
vch = [ch.array(v[mesh]) for mesh in rangeMeshes]
if len(vch)==1:
    vstack = vch[0]
else:
    vstack = ch.vstack(vch)

rangeMeshes = range(len(vmod))
vchmod = [ch.array(vmod[mesh]) for mesh in rangeMeshes]
if len(vchmod)==1:
    vstackmod = vchmod[0]
else:
    vstackmod = ch.vstack(vchmod)

# GT Azimuth: [ 1.30899694]
# Azimuth: [ 0.78539816]
# GT Elevation: [ 0.52359878]
# Elevation: [ 0.87266463]

# chAz = ch.Ch([radians(azimuth)])
# chEl = ch.Ch([radians(elevation)])
chAz = ch.Ch([4.742895587179587])
chEl = ch.Ch([0.22173048])
chDist = ch.Ch([camDistance])

# chAzGT = ch.Ch([radians(azimuth)])
# chElGT = ch.Ch([radians(elevation)])

# chAzGT = ch.Ch([1.30899694])
chAzGT = ch.Ch([4.742895587179587])
chElGT = ch.Ch([0.22120681])
chDistGT = ch.Ch([camDistance])

cameraGT, modelRotationGT = setupCamera(vstack, chAzGT, chElGT, chDistGT, center, width, height)

camera, modelRotation = setupCamera(vstackmod, chAz, chEl, chDist, center, width, height)

rangeMeshes = range(len(v))

vnch = [ch.transpose(ch.dot(modelRotationGT, ch.transpose(ch.array(vn[mesh])))) for mesh in rangeMeshes]
vcch = [ch.array(vc[mesh]) for mesh in rangeMeshes]

light_color=ch.ones(3)
chComponentGT = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))
chComponent = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))

A_list = [SphericalHarmonics(vn=vnch[mesh],
                       components=chComponentGT,
                       light_color=light_color) for mesh in rangeMeshes]

vc_list = [A_list[mesh]*vcch[mesh] for mesh in rangeMeshes]

rangeMeshes = range(len(vmod))

vnchmod = [ch.transpose(ch.dot(modelRotation, ch.transpose(ch.array(vnmod[mesh])))) for mesh in rangeMeshes]
vcchmod = [ch.array(vcmod[mesh]) for mesh in rangeMeshes]

light_color=ch.ones(3)
chComponent = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))

Amod_list = [SphericalHarmonics(vn=vnchmod[mesh],
                       components=chComponent,
                       light_color=light_color) for mesh in rangeMeshes]

vcmod_list = [Amod_list[mesh]*vcch[mesh] for mesh in rangeMeshes]



# ipdb.set_trace()

frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}


rendererGT = setupTexturedRenderer(vstack, vch, f_list, vc_list, vnch,  uv, haveTextures_list, textures_list, cameraGT, frustum)

renderer = setupTexturedRenderer(vstackmod, vchmod, fmod_list, vcmod_list, vnchmod,  uvmod, haveTexturesmod_list, texturesmod_list, camera, frustum)

vis_gt = np.array(rendererGT.image_mesh_bool(0)).copy().astype(np.bool)
vis_mask = np.array(rendererGT.indices_image==1).copy().astype(np.bool)
vis_im = np.array(renderer.image_mesh_bool(0)).copy().astype(np.bool)

oldChAz = chAz[0].r
oldChEl = chEl[0].r

# Show it
shapeIm = vis_gt.shape
numPixels = shapeIm[0] * shapeIm[1]
shapeIm3D = [vis_im.shape[0], vis_im.shape[1], 3]

print("Beginning render.")
t = time.time()
rendererGT.r
elapsed_time = time.time() - t
print("Ended render in  " + str(elapsed_time))
plt.imsave('opendr_opengl_first.png', rendererGT.r)

imagegt = np.copy(np.array(rendererGT.r)).astype(np.float64)
chImage = ch.array(imagegt)
# E_raw_simple = renderer - rendererGT
negVisGT = ~vis_gt
imageWhiteMask = imagegt.copy()
imageWhiteMask[np.tile(negVisGT.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1

chImageWhite = ch.Ch(imageWhiteMask)
E_raw = renderer - rendererGT
SE_raw = ch.sum(E_raw*E_raw, axis=2)


SSqE_raw = ch.SumOfSquares(E_raw)/numPixels

stds = ch.Ch([0.1])
variances = stds ** 2
globalPrior = ch.Ch([0.8])

negLikModel = -score_image.modelLogLikelihoodCh(rendererGT, renderer, vis_im, 'FULL', variances)/numPixels

negLikModelRobust = -score_image.modelLogLikelihoodRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances)/numPixels

pixelLikelihoodCh = score_image.logPixelLikelihoodCh(rendererGT, renderer, vis_im, 'FULL', variances)

pixelLikelihoodRobustCh = ch.log(score_image.pixelLikelihoodRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances))

post = score_image.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances)[0]

models = [negLikModel, negLikModelRobust]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh]
modelsDescr = ["Gaussian Model", "Outlier model"]
# , negLikModelPyr, negLikModelRobustPyr, SSqE_raw

global model
model = 0

pixelErrorFun = pixelLikelihoodCh
errorFun = negLikModel

iterat = 0

f, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'aspect':'equal'}, figsize=(9, 12))
pos1 = ax1.get_position()
pos5 = ax5.get_position()
pos5.x0 = pos1.x0
ax5.set_position(pos5)

f.tight_layout()

ax1.set_title("Ground Truth")

ax2.set_title("Backprojection")
pim2 = ax2.imshow(renderer.r)

edges = renderer.boundarybool_image
gtoverlay = imagegt.copy()
gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
pim1 = ax1.imshow(gtoverlay)

ax3.set_title("Pixel negative log probabilities")
pim3 = ax3.imshow(-pixelErrorFun.r)
cb3 = plt.colorbar(pim3, ax=ax3,use_gridspec=True)
cb3.mappable = pim3

ax4.set_title("Posterior probabilities")
ax4.imshow(np.tile(post.reshape(shapeIm[0],shapeIm[1],1), [1,1,3]))

ax5.set_title("Dr wrt. Azimuth")
drazsum = -pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
img5 = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
cb5 = plt.colorbar(img5, ax=ax5,use_gridspec=True)
cb5.mappable = img5

ax6.set_title("Dr wrt. Elevation")
drazsum = -pixelErrorFun.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
img6 = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
cb6 = plt.colorbar(img6, ax=ax6,use_gridspec=True)
cb6.mappable = img6

pos1 = ax1.get_position()
pos5 = ax5.get_position()
pos5.x0 = pos1.x0
ax5.set_position(pos5)

pos1 = ax1.get_position()
pos5 = ax5.get_position()
pos5.x0 = pos1.x0
ax5.set_position(pos5)

plt.show()
plt.pause(0.1)

elapsed_time = time.time() - t
global changedGT
changedGT = False
global refresh
refresh = True
global drawSurf
drawSurf = False
global makeVideo
makeVideo = False

if makeVideo:
    import matplotlib.animation as animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(title='', artist=''), bitrate=1800)
    figvid, (vax1, vax2) = plt.subplots(1, 2, sharey=True, subplot_kw={'aspect':'equal'}, figsize=(12, 6))
    vax1.axes.get_xaxis().set_visible(False)
    vax1.axes.get_yaxis().set_visible(False)
    vax1.set_title("Ground truth")
    vax2.axes.get_xaxis().set_visible(False)
    vax2.axes.get_yaxis().set_visible(False)
    vax2.set_title("Backprojection")

    plt.tight_layout()

computePerformance = True
performance = {}
elevations = {}
azimuths = {}
gradEl = {}
gradAz = {}
performanceSurf = {}
elevationsSurf = {}
azimuthsSurf = {}
gradElSurf = {}
gradAzSurf = {}
gradFinElSurf = {}
gradFinAzSurf = {}

if computePerformance:
    from mpl_toolkits.mplot3d import Axes3D
    global figperf
    figperf = plt.figure()
    global axperf
    axperf = figperf.add_subplot(111, projection='3d')
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    # x1,x2,y1,y2 = plt.axis()
    # plt.axis((0,360,0,90))
    performance[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    azimuths[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    elevations[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    gradAz[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    gradEl[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

    performanceSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

    gradElSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

    gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

ims = []

def refreshSubplots():
        #Other subplots visualizing renders and its pixel derivatives
        edges = renderer.boundarybool_image
        imagegt = np.copy(np.array(rendererGT.r)).astype(np.float64)
        gtoverlay = imagegt.copy()
        gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
        pim1.set_data(gtoverlay)
        pim2.set_data(renderer.r)
        pim3 = ax3.imshow(-pixelErrorFun.r)
        cb3.mappable = pim3
        cb3.update_normal(pim3)
        ax4.imshow(np.tile(post.reshape(shapeIm[0],shapeIm[1],1), [1,1,3]))
        drazsum = -pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
        img5 = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
        cb5.mappable = img5
        cb5.update_normal(img5)
        drazsum = -pixelErrorFun.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
        img6 = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
        cb6.mappable = img6
        cb6.update_normal(img6)
        f.canvas.draw()
        plt.pause(0.1)


def plotSurface():
    global figperf
    global axperf
    global surf
    global line
    global drawSurf
    global computePerformance
    global plotMinimization
    global chAz
    global chEl
    global model
    if not plotMinimization and not drawSurf:
        figperf.clear()
        global axperf
        axperf = figperf.add_subplot(111, projection='3d')

    plt.figure(figperf.number)
    axperf.clear()
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    scaleSurfGrads = 0
    if drawSurf:
        from scipy.interpolate import griddata
        x1 = np.linspace((azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).min(), (azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).max(), len((azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi)))
        y1 = np.linspace((elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).min(), (elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).max(), len((elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi)))
        x2, y2 = np.meshgrid(x1, y1)
        z2 = griddata(((azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi), (elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi)), performanceSurf[(model, chAzGT.r[0], chElGT.r[0])], (x2, y2), method='cubic')
        from matplotlib import cm, colors
        surf = axperf.plot_surface(x2, y2, z2, rstride=3, cstride=3, cmap=cm.coolwarm, linewidth=0.1, alpha=0.85)
        avgSurfGradMagnitudes = np.mean(np.sqrt(gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])]**2+gradElSurf[(model, chAzGT.r[0], chElGT.r[0])]**2))
        scaleSurfGrads = 5./avgSurfGradMagnitudes
        for point in range(len(performanceSurf[(model, chAzGT.r[0], chElGT.r[0])])):
            perfi = performanceSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
            azi = azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
            eli = elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
            gradAzi = -gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
            gradEli = -gradElSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
            arrowGrad = Arrow3D([azi*180./np.pi, azi*180./np.pi + gradAzi*scaleSurfGrads], [eli*180./np.pi, eli*180./np.pi + gradEli*scaleSurfGrads], [perfi, perfi], mutation_scale=10, lw=1, arrowstyle="-|>", color="b")
            axperf.add_artist(arrowGrad)

            diffAzi = -gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
            diffEli = -gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
            colorArrow = 'g'
            if diffAzi * gradAzi + diffEli * gradEli < 0:
                colorArrow = 'r'
            arrowGradDiff = Arrow3D([azi*180./np.pi, azi*180./np.pi + diffAzi*scaleSurfGrads], [eli*180./np.pi, eli*180./np.pi + diffEli*scaleSurfGrads], [perfi, perfi], mutation_scale=10, lw=1, arrowstyle="-|>", color=colorArrow)
            axperf.add_artist(arrowGradDiff)

        axperf.plot([chAzGT.r[0]*180./np.pi, chAzGT.r[0]*180./np.pi], [chElGT.r[0]*180./np.pi,chElGT.r[0]*180./np.pi], [z2.min(), z2.max()], 'b--', linewidth=1)

        axperf.plot(chAz.r*180./np.pi, chEl.r*180./np.pi, errorFun.r[0], 'yD')

        import scipy.sparse as sp
        if sp.issparse(errorFun.dr_wrt(chAz)):
            drAz = -errorFun.dr_wrt(chAz).toarray()[0][0]
        else:
            drAz = -errorFun.dr_wrt(chAz)[0][0]
        if sp.issparse(errorFun.dr_wrt(chEl)):
            drEl = -errorFun.dr_wrt(chEl).toarray()[0][0]
        else:
            drEl = -errorFun.dr_wrt(chEl)[0][0]

        chAzOldi = chAz.r[0]
        chElOldi = chEl.r[0]
        diffAz = -ch.optimization.gradCheckSimple(errorFun, chAz, 0.01745)
        diffEl = -ch.optimization.gradCheckSimple(errorFun, chEl, 0.01745)
        chAz[0] = chAzOldi
        chEl[0] = chElOldi

        arrowGrad = Arrow3D([chAz.r[0]*180./np.pi, chAz.r[0]*180./np.pi + drAz*scaleSurfGrads], [chEl.r[0]*180./np.pi, chEl.r[0]*180./np.pi + drEl*scaleSurfGrads], [errorFun.r[0], errorFun.r[0]], mutation_scale=10, lw=1, arrowstyle="-|>", color="b")
        axperf.add_artist(arrowGrad)
        colorArrow = 'g'
        if diffAz * drAz + diffEl * drEl < 0:
            colorArrow = 'r'
        arrowGradDiff = Arrow3D([chAz.r[0]*180./np.pi, chAz.r[0]*180./np.pi + diffAz*scaleSurfGrads], [chEl.r[0]*180./np.pi, chEl.r[0]*180./np.pi + diffEl*scaleSurfGrads], [errorFun.r[0], errorFun.r[0]], mutation_scale=10, lw=1, arrowstyle="-|>", color=colorArrow)
        axperf.add_artist(arrowGradDiff)

    if plotMinimization:
        if azimuths.get((model, chAzGT.r[0], chElGT.r[0])) != None:
            axperf.plot(azimuths[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, elevations[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, performance[(model, chAzGT.r[0], chElGT.r[0])], color='g', linewidth=1.5)
            axperf.plot(azimuths[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, elevations[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, performance[(model, chAzGT.r[0], chElGT.r[0])], 'rD')

    axperf.set_xlabel('Azimuth (degrees)')
    axperf.set_ylabel('Elevation (degrees)')
    axperf.set_zlabel('Negative Log Likelihood')
    plt.title('Model type: ' + str(model))

    plt.pause(0.1)
    plt.draw()

def cb2(_):
    global t
    elapsed_time = time.time() - t
    print("Ended interation in  " + str(elapsed_time))

    global pixelErrorFun
    global errorFun
    global iterat
    iterat = iterat + 1
    print("Callback! " + str(iterat))
    print("Sq Error: " + str(errorFun.r))
    global imagegt
    global renderer
    global gradAz
    global gradEl
    global performance
    global azimuths
    global elevations

    refreshSubplots()

    if makeVideo:
        plt.figure(figvid.number)
        im1 = vax1.imshow(gtoverlay)

        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)

        t = vax1.annotate("Minimization iteration: " + str(iterat), xy=(1, 0), xycoords='axes fraction', fontsize=16,
                    xytext=(-20, 5), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props)
        im2 = vax2.imshow(renderer.r)
        ims.append([im1, im2, t])

    if computePerformance:
        if performance.get((model, chAzGT.r[0], chElGT.r[0])) == None:
            performance[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            azimuths[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            elevations[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradAz[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradEl[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        performance[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(performance[(model, chAzGT.r[0], chElGT.r[0])], errorFun.r)
        azimuths[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(azimuths[(model, chAzGT.r[0], chElGT.r[0])], chAz.r)
        elevations[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(elevations[(model, chAzGT.r[0], chElGT.r[0])], chEl.r)

        import scipy.sparse as sp
        if sp.issparse(errorFun.dr_wrt(chAz)):
            drAz = errorFun.dr_wrt(chAz).toarray()[0][0]
        else:
            drAz = errorFun.dr_wrt(chAz)[0][0]
        if sp.issparse(errorFun.dr_wrt(chEl)):
            drEl = errorFun.dr_wrt(chEl).toarray()[0][0]
        else:
            drEl = errorFun.dr_wrt(chEl)[0][0]

        gradAz[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradAz[(model, chAzGT.r[0], chElGT.r[0])], drAz)
        gradEl[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradEl[(model, chAzGT.r[0], chElGT.r[0])], drEl)

        plotSurface()


    plt.pause(0.1)
    plt.show()
    plt.draw()
    plt.pause(0.1)
    t = time.time()

# , chComponent[0]
free_variables = [chAz, chEl]

mintime = time.time()
boundEl = (0, np.pi/2.0)
boundAz = (0, None)
boundscomponents = (0,None)
bounds = [boundAz,boundEl]
methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead']
method = 1
exit = False
minimize = False
plotMinimization = False
global chAzSaved
global chElSaved
global chComponentSaved
chAzSaved = chAz.r[0]
chElSaved = chEl.r[0]
chComponentSaved = chComponent.r[0]

def readKeys(window, key, scancode, action, mods):
    print("Reading keys...")
    global exit
    global refresh
    global chAz
    global chEl
    global chComponent
    refresh = False
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_ESCAPE and action == glfw.RELEASE:
        glfw.set_window_should_close(window, True)
        exit = True
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_LEFT and action == glfw.RELEASE:
        refresh = True
        chAz[0] = chAz[0].r - radians(5)
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_RIGHT and action == glfw.RELEASE:
        refresh = True
        chAz[0] = chAz[0].r + radians(5)
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_DOWN and action == glfw.RELEASE:
        refresh = True
        chEl[0] = chEl[0].r - radians(5)
        refresh = True
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_UP and action == glfw.RELEASE:
        refresh = True
        chEl[0] = chEl[0].r + radians(5)
    if mods==glfw.MOD_SHIFT and key == glfw.KEY_LEFT and action == glfw.RELEASE:
        print("Left modifier!")
        refresh = True
        chAz[0] = chAz[0].r - radians(1)
    if mods==glfw.MOD_SHIFT and key == glfw.KEY_RIGHT and action == glfw.RELEASE:
        refresh = True
        chAz[0] = chAz[0].r + radians(1)
    if mods==glfw.MOD_SHIFT and key == glfw.KEY_DOWN and action == glfw.RELEASE:
        refresh = True
        chEl[0] = chEl[0].r - radians(1)
        refresh = True
    if mods==glfw.MOD_SHIFT and key == glfw.KEY_UP and action == glfw.RELEASE:
        refresh = True
        chEl[0] = chEl[0].r + radians(1)
    if key == glfw.KEY_C and action == glfw.RELEASE:
        print("Grad check: " + ch.optimization.gradCheck(errorFun, [chAz], [0.01745]))
        print("Scipy grad check: " + ch.optimization.scipyGradCheck({'raw': errorFun}, [chAz]))
    if key == glfw.KEY_B:
        refresh = True
        chComponent[0] = chComponent[0].r + 0.1
    if key == glfw.KEY_D:
        refresh = True
        chComponent[0] = chComponent[0].r - 0.1
    global changedGT
    global drawSurf
    global model
    global models
    if key == glfw.KEY_G and action == glfw.RELEASE:
        refresh = True
        changedGT = True

    if key == glfw.KEY_E and action == glfw.RELEASE:
        if computePerformance:
            print("Estimating cost function surface and gradients...")
            drawSurf = True
            chAzOld = chAz.r[0]
            chElOld = chEl.r[0]

            for model_num, errorFun in enumerate(models):
                performanceSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])
                azimuthsSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])
                elevationsSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])

                gradAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])
                gradElSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])

                gradFinAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])
                gradFinElSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])

            for chAzi in np.linspace(max(chAzGT.r[0]-np.pi/8.,0), min(chAzGT.r[0] + np.pi/8., 2.*np.pi), num=10):
                for chEli in np.linspace(max(chElGT.r[0]-np.pi/8,0), min(chElGT.r[0]+np.pi/8, np.pi/2), num=10):
                    for model_num, errorFun in enumerate(models):
                        chAz[:] = chAzi
                        chEl[:] = chEli

                        performanceSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(performanceSurf[(model_num, chAzGT.r[0], chElGT.r[0])], errorFun.r)
                        azimuthsSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(azimuthsSurf[(model_num, chAzGT.r[0], chElGT.r[0])], chAzi)
                        elevationsSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(elevationsSurf[(model_num, chAzGT.r[0], chElGT.r[0])], chEli)
                        import scipy.sparse as sp
                        if sp.issparse(errorFun.dr_wrt(chAz)):
                            drAz = errorFun.dr_wrt(chAz).toarray()[0][0]
                        else:
                            drAz = errorFun.dr_wrt(chAz)[0][0]
                        if sp.issparse(errorFun.dr_wrt(chEl)):
                            drEl = errorFun.dr_wrt(chEl).toarray()[0][0]
                        else:
                            drEl = errorFun.dr_wrt(chEl)[0][0]

                        gradAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])], drAz)
                        gradElSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradElSurf[(model_num, chAzGT.r[0], chElGT.r[0])], drEl)
                        chAzOldi = chAz.r[0]
                        chElOldi = chEl.r[0]
                        diffAz = ch.optimization.gradCheckSimple(errorFun, chAz, 0.01745)
                        diffEl = ch.optimization.gradCheckSimple(errorFun, chEl, 0.01745)
                        chAz[:] = chAzOldi
                        chEl[:] = chElOldi
                        gradFinAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradFinAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])], diffAz)
                        gradFinElSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradFinElSurf[(model_num, chAzGT.r[0], chElGT.r[0])], diffEl)
            errorFun = models[model]

            chAz[:] = chAzOld
            chEl[:] = chElOld

            refresh = True
            print("Finshed estimating.")

    if key == glfw.KEY_P and action == glfw.RELEASE:
        ipdb.set_trace()
        refresh = True

    if key == glfw.KEY_N and action == glfw.RELEASE:
        print("Back to GT!")
        chAz[:] = chAzGT.r[:]
        chEl[:] = chElGT.r[:]
        chComponent[:] = chComponentGT.r[:]
        refresh = True

    global chAzSaved
    global chElSaved
    global chComponentSaved

    if key == glfw.KEY_Z and action == glfw.RELEASE:
        print("Saved!")
        chAzSaved = chAz.r[0]
        chElSaved = chEl.r[0]
        chComponentSaved = chComponent.r[0]

    if key == glfw.KEY_X and action == glfw.RELEASE:
        print("Back to Saved!")
        chAz[0] = chAzSaved
        chEl[0] = chElSaved
        chComponent[0] = chComponentSaved
        refresh = True

    if key == glfw.KEY_S and action == glfw.RELEASE:
        print("**** Statistics ****" )
        print("GT Azimuth: " + str(chAzGT))
        print("Azimuth: " + str(chAz))
        print("GT Elevation: " + str(chElGT))
        print("Elevation: " + str(chEl))

        print("Dr wrt Azimuth: " + str(errorFun.dr_wrt(chAz)))
        print("Dr wrt Elevation: " + str(errorFun.dr_wrt(chEl)))
        # print("Dr wrt Distance: " + str(errorFun.dr_wrt(chDist)))

        print("Occlusion is " + str(getOcclusionFraction(rendererGT)*100) + " %")

        if drawSurf:
            avgError = np.mean(np.sqrt((gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] - gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])])**2 + (gradElSurf[(model, chAzGT.r[0], chElGT.r[0])] - gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])])**2))
            print("** Approx gradients - finite differenes." )
            print("Avg Eucl. distance :: " + str(avgError))
            norm2Grad = np.sqrt((gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])])**2 + (gradElSurf[(model, chAzGT.r[0], chElGT.r[0])])**2)
            norm2Diff = np.sqrt((gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])])**2 + (gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])])**2)
            avgAngle = np.arccos((gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] + gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradElSurf[(model, chAzGT.r[0], chElGT.r[0])])/(norm2Grad*norm2Diff))
            print("Avg Angle.: " + str(np.mean(avgAngle)))
            print("Num opposite (red) gradients: " + str(np.sum((gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] + gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradElSurf[(model, chAzGT.r[0], chElGT.r[0])]) < 0)))
            idxmin = np.argmin(performanceSurf[(model, chAzGT.r[0], chElGT.r[0])])
            # distGTMin = np.sqrt((chAzGT*180/np.pi - azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]*180/np.pi)**2 + (chAzGT*180/np.pi-elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]*180/np.pi)**2)
            # print("Dist of minimum to groundtruth: " + str(distGTMin))
            azDiff = np.arctan2(np.arcsin(chAzGT - azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]), np.arccos(chAzGT - azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]))
            elDiff = np.arctan2(np.arcsin(chElGT - elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]), np.arccos(chElGT - elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]))
            print("Minimum Azimuth difference of " + str(azDiff*180/np.pi))
            print("Minimum Elevation difference of " + str(elDiff*180/np.pi))

        azDiff = np.arctan2(np.arcsin(chAzGT - chAz.r[0]), np.arccos(chAzGT - chAz.r[0]))
        elDiff = np.arctan2(np.arcsin(chElGT - chEl.r[0]), np.arccos(chElGT - chEl.r[0]))
        print("Current Azimuth difference of " + str(azDiff*180/np.pi))
        print("Current Elevation difference of " + str(elDiff*180/np.pi))

    if key == glfw.KEY_V and action == glfw.RELEASE:
        global ims
        if makeVideo:
            im_ani = animation.ArtistAnimation(figvid, ims, interval=2000, repeat_delay=3000, repeat=False, blit=True)
            im_ani.save('minimization_demo.mp4', fps=None, writer=writer, codec='mp4')
            ims = []

    global stds
    global globalPrior
    global plotMinimization
    if key == glfw.KEY_KP_1 and action == glfw.RELEASE:
        stds[:] = stds.r[0]/1.5
        print("New standard devs of " + str(stds.r))
        refresh = True
        drawSurf = False
        plotMinimization = False
    if key == glfw.KEY_KP_2 and action == glfw.RELEASE:
        stds[:] = stds.r[0]*1.5
        print("New standard devs of " + str(stds.r))
        refresh = True
        drawSurf = False
        plotMinimization = False

    if key == glfw.KEY_KP_4 and action == glfw.RELEASE:
        globalPrior[0] = globalPrior.r[0] - 0.05
        print("New foreground prior of" + str(globalPrior.r))
        refresh = True
        drawSurf = False
        plotMinimization = False
    if key == glfw.KEY_KP_5 and action == glfw.RELEASE:
        globalPrior[0] = globalPrior.r[0] + 0.05
        print("New foreground prior of " + str(globalPrior.r))
        refresh = True
        drawSurf = False
        plotMinimization = False

    import regression_methods
    global azsGT
    global elevsGT
    global split
    global randForestModel
    global linRegModel
    global numDataset
    import imageproc

    if key == glfw.KEY_T and action == glfw.RELEASE:

        ipdb.set_trace()
        numDataset = 10
        chAzOld = chAz.r[0]
        chElOld = chEl.r[0]
        chAzGTOld = chAzGT.r[0]
        chElGTOld = chElGT.r[0]

        azsGT = numpy.random.uniform(4.742895587179587 - np.pi/4,4.742895587179587 + np.pi/4, numDataset)
        elevsGT = numpy.random.uniform(1,np.pi/2, numDataset)
        images = []
        occlusions = []
        hogs = []

        split = 0.8
        setTrain = np.arange(np.floor(len(azsGT)*split)).astype(np.uint8)
        for train_i in setTrain:
            azi = azsGT[train_i]
            eli = elevsGT[train_i]
            chAzGT[:] = azi
            chElGT[:] = eli
            image = rendererGT.r.copy()
            images = images + [image]
            occlusions = occlusions + [getOcclusionFraction(rendererGT)]
            hogs = hogs + [imageproc.computeHoG(image).reshape([1,-1])]

        hogfeats = np.vstack(hogs)

        #Train
        randForestModelCosAzs = regression_methods.trainRandomForest(hogfeats, np.cos(azsGT[setTrain]))
        randForestModelSinAzs = regression_methods.trainRandomForest(hogfeats, np.sin(elevsGT[setTrain]))
        linRegModelCosAzs = regression_methods.trainLinearRegression(hogfeats, np.cos(azsGT[setTrain]))
        linRegModelSinAzs = regression_methods.trainLinearRegression(hogfeats, np.sin(elevsGT[setTrain]))

        chAz[:] = chAzOld
        chEl[:] = chElOld
        chAzGT[:] = chAzGTOld
        chElGT[:] = chElGTOld

    if key == glfw.KEY_I and action == glfw.RELEASE:

        elevsPred = np.arctan2(np.sin(elevsPred), np.cos(elevsPred))
        azsPred = np.arctan2(np.sin(azsPred), np.cos(azsPred))

        randForestModelCosAzs
        randForestModelSinAzs
        linRegModelCosAzs
        linRegModelSinAzs

        evaluatePrediction(azsGT, elevsGT, azsPred, elevsPred)

    if key == glfw.KEY_R and action == glfw.RELEASE:
        global errorFun
    global pixelErrorFun
    global model
    global models
    global modelsDescr
    global pixelModels

    if key == glfw.KEY_O and action == glfw.RELEASE:
        # drawSurf = False
        model = (model + 1) % len(models)
        print("Using " + modelsDescr[model])
        errorFun = models[model]
        pixelErrorFun = pixelModels[model]

        refresh = True

    global method
    global methods
    if key == glfw.KEY_1 and action == glfw.RELEASE:
        method = 0
        print("Changed to minimizer: " + methods[method])
    if key == glfw.KEY_2 and action == glfw.RELEASE:
        method = 1
        print("Changed to minimizer: " + methods[method])
    if key == glfw.KEY_3 and action == glfw.RELEASE:
        method = 2
        print("Changed to minimizer: " + methods[method])
    if key == glfw.KEY_4 and action == glfw.RELEASE:
        print("Changed to minimizer: " + methods[method])
        method = 3
    if key == glfw.KEY_5 and action == glfw.RELEASE:
        method = 4
        print("Changed to minimizer: " + methods[method])

    global minimize
    if key == glfw.KEY_M and action == glfw.RELEASE:
        minimize = True

glfw.make_context_current(renderer.win)

glfw.set_key_callback(renderer.win, readKeys)

while not exit:
    # Poll for and process events
    glfw.make_context_current(renderer.win)
    glfw.poll_events()

    if changedGT:
        drawSurf = False
        plotMinimization = False
        imagegt = np.copy(np.array(rendererGT.r)).astype(np.float64)
        chImage[:,:,:] = imagegt[:,:,:]

        chAzGT[:] = chAz.r[:]
        chElGT[:] = chEl.r[:]
        chDistGT[:] = chDist.r[:]
        chComponentGT[:] = chComponent.r[:]

        changedGT = False

        if makeVideo:
            ims = []

        performance[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        azimuths[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        elevations[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        gradAz[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        gradEl[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

        performanceSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

        gradElSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

        gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

    if refresh:
        print("Sq Error: " + str(errorFun.r))

        refreshSubplots()


        if computePerformance and drawSurf:
            plotSurface()

        plt.pause(0.1)
        plt.draw()
        refresh = False

    if minimize:
        iterat = 0
        print("Minimizing with method " + methods[method])
        ch.minimize({'raw': errorFun}, bounds=bounds, method=methods[method], x0=free_variables, callback=cb2, options={'disp':True})
        plotMinimization = True
        minimize = False

refreshSubplots()

plt.imsave('opendr_opengl_final.png', rendererGT.r)
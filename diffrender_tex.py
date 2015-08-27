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

width, height = (100, 100)

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

chAz = ch.Ch([radians(azimuth)])
chEl = ch.Ch([radians(elevation)])
chDist = ch.Ch([camDistance])

chAzGT = ch.Ch([radians(azimuth)])
chElGT = ch.Ch([radians(elevation)])
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

f, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'aspect':'equal'}, figsize=(9, 12))
pos1 = ax1.get_position()
pos5 = ax5.get_position()
pos5.x0 = pos1.x0
ax5.set_position(pos5)

f.tight_layout()

ax1.set_title("Ground Truth")
ax1.imshow(rendererGT.r)

plt.imsave('opendr_opengl_gt.png', rendererGT.r)
plt.draw()

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

# E_pyr = gaussian_pyramid(E_raw, n_levels=4, normalization='SSE')
# E_pyr_simple = gaussian_pyramid(E_raw_simple, n_levels=4, normalization='SSE')

SSqE_raw = ch.SumOfSquares(E_raw)/numPixels
# ch.SumOfSquares(E_raw)/np.sum(vis_im)
# SSqE_raw_simple = ch.SumOfSquares(E_raw_simple)/vis_im.size
# SSqE_pyr = ch.SumOfSquares(E_pyr_simple)/vis_im.size
# ch.SumOfSquares(E_pyr)/np.sum(vis_im)
# SSqE_pyr_simple = ch.SumOfSquares(E_pyr_simple)/vis_im.size

variances = (numpy.ones(shapeIm3D)*25.0/255.0) ** 2
globalPrior = 0.9

negLikModel = -score_image.modelLogLikelihoodCh(rendererGT, renderer, vis_im, 'FULL', variances)/numPixels

negLikModelRobust = -score_image.modelLogLikelihoodRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances)/numPixels

pixelLikelihoodCh = -score_image.logPixelLikelihoodCh(rendererGT, renderer, vis_im, 'FULL', variances)
negLikModelPyr = gaussian_pyramid(pixelLikelihoodCh, n_levels=4, normalization='SSE')

pixelLikelihoodRobustCh = -ch.log(score_image.pixelLikelihoodRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances))
# pixelLikelihoodRobustCh2 = -ch.log(score_image.pixelLikelihoodCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances))

post = score_image.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances)[0]

# pixelErrorFun = S
# errorFun = negLikModel
global model
model = 2
pixelErrorFun = SE_raw
errorFun = SSqE_raw

iterat = 0

ax2.set_title("Backprojection")
pim2 = ax2.imshow(renderer.r)

plt.draw()

edges = renderer.boundarybool_image
gtoverlay = imagegt.copy()
gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
pim1 = ax1.imshow(gtoverlay)

ax3.set_title("Error (Abs of residuals)")
pim3 = ax3.imshow(pixelErrorFun.r)

ax4.set_title("Posterior probabilities")
ax4.imshow(np.tile(post.reshape(shapeIm[0],shapeIm[1],1), [1,1,3]))

ax5.set_title("Dr wrt. Azimuth")
drazsum = pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
img = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
plt.colorbar(img, ax=ax5,use_gridspec=True)

ax6.set_title("Dr wrt. Elevation")
drazsum = pixelErrorFun.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
img6 = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
plt.colorbar(img6, ax=ax6,use_gridspec=True)

pos1 = ax1.get_position()
pos5 = ax5.get_position()
pos5.x0 = pos1.x0
ax5.set_position(pos5)

plt.show()
pos1 = ax1.get_position()
pos5 = ax5.get_position()
pos5.x0 = pos1.x0
ax5.set_position(pos5)

plt.pause(0.1)

elapsed_time = time.time() - t
changedGT = False
refresh = True

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
performanceSurf = {}
elevationsSurf = {}
azimuthsSurf = {}

if computePerformance:
    from mpl_toolkits.mplot3d import Axes3D
    figperf = plt.figure()
    axperf = figperf.add_subplot(111, projection='3d')
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    # x1,x2,y1,y2 = plt.axis()
    # plt.axis((0,360,0,90))
    performance[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    azimuths[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    elevations[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

    performanceSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

ims = []
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

    edges = renderer.boundarybool_image
    gtoverlay = imagegt.copy()
    gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1

    if makeVideo:
        plt.figure(figvid.number)
        im1 = vax1.imshow(gtoverlay)

        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)

        t = vax1.annotate("Minimization iteration: " + str(iterat), xy=(1, 0), xycoords='axes fraction', fontsize=16,
                    xytext=(-20, 5), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props)

        # figvid.suptitle()

        im2 = vax2.imshow(renderer.r)
        ims.append([im1, im2, t])

    if computePerformance:
        performance[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(performance[(model, chAzGT.r[0], chElGT.r[0])], errorFun.r)
        azimuths[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(azimuths[(model, chAzGT.r[0], chElGT.r[0])], chAz.r)
        elevations[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(elevations[(model, chAzGT.r[0], chElGT.r[0])], chEl.r)
        global figperf
        global axperf
        global surf
        global line
        plt.figure(figperf.number)
        axperf.clear()
        from matplotlib.font_manager import FontProperties
        fontP = FontProperties()
        fontP.set_size('small')

        performanceSurf[(model, chAzGT.r[0], chElGT.r[0])]

            # try:
        # line.remove()
            # except:    #     print("no line")

        from scipy.interpolate import griddata
        x1 = np.linspace((azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).min(), (azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).max(), len((azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi)))
        y1 = np.linspace((elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).min(), (elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).max(), len((elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi)))
        x2, y2 = np.meshgrid(x1, y1)
        z2 = griddata(((azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi), (elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi)), performanceSurf[(model, chAzGT.r[0], chElGT.r[0])], (x2, y2), method='cubic')
        from matplotlib import cm, colors
        surf = axperf.plot_surface(x2, y2, z2, rstride=3, cstride=3, cmap=cm.coolwarm, linewidth=0.1, alpha=0.85)


        # plt.axvline(x=bestAzimuth, linewidth=2, color='b', label='Minimum score azimuth')
        # plt.axvline(x=chAzGT, linewidth=2, color='g', label='Ground truth azimuth')
        # plt.axvline(x=(bestAzimuth + 180) % 360, linewidth=1, color='b', ls='--', label='Minimum distance azimuth + 180')

        # fig.savefig(numDir + 'performance.png')

        line = axperf.plot(azimuths[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, elevations[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, performance[(model, chAzGT.r[0], chElGT.r[0])], color='g', linewidth=1.5)
        line = axperf.plot(azimuths[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, elevations[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, performance[(model, chAzGT.r[0], chElGT.r[0])], 'rD')

        axperf.set_xlabel('Azimuth (degrees)')
        axperf.set_ylabel('Elevation (degrees)')
        axperf.set_zlabel('Negative Log Likelihood')
        plt.title('Model type: ' + str(model))




    pim1.set_data(gtoverlay)
    pim2.set_data(renderer.r)
    pim3 = ax3.imshow(pixelErrorFun.r)
    ax4.set_title("Posterior probabilities")
    ax4.imshow(np.tile(post.reshape(shapeIm[0],shapeIm[1],1), [1,1,3]))
    drazsum = pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
    img = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    drazsum = pixelErrorFun.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
    img = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    f.canvas.draw()
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


def readKeys(window, key, scancode, action, mods):
    print("Reading keys...")
    global exit
    global refresh
    refresh = False
    if key == glfw.KEY_ESCAPE and action == glfw.RELEASE:
        glfw.set_window_should_close(window, True)
        exit = True
    if key == glfw.KEY_LEFT:
        refresh = True
        chAz[0] = chAz[0].r - radians(5)
    if key == glfw.KEY_RIGHT:
        refresh = True
        chAz[0] = chAz[0].r + radians(5)
    if key == glfw.KEY_DOWN:
        refresh = True
        chEl[0] = chEl[0].r - radians(5)
        refresh = True
    if key == glfw.KEY_UP:
        refresh = True
        chEl[0] = chEl[0].r + radians(5)
    if key == glfw.KEY_C and action == glfw.RELEASE:
        print("Grad check: " + ch.optimization.gradCheck(errorFun, [chAz], [0.01]))
        print("Scipy grad check: " + ch.optimization.scipyGradCheck({'raw': errorFun}, [chAz]))
    if key == glfw.KEY_B:
        refresh = True
        chComponent[0] = chComponent[0].r + 0.1
    if key == glfw.KEY_D:
        refresh = True
        chComponent[0] = chComponent[0].r - 0.1
    global changedGT
    if key == glfw.KEY_G and action == glfw.RELEASE:
        refresh = True
        changedGT = True

    if key == glfw.KEY_E and action == glfw.RELEASE:
        chAzOld = chAz.r[0]
        chElOld = chEl.r[0]
        for chAzi in np.linspace(max(chAzGT.r[0]-np.pi/8.,0), min(chAzGT.r[0] + np.pi/8., 2.*np.pi), num=10):
            for chEli in np.linspace(max(chElGT.r[0]-np.pi/8,0), min(chElGT.r[0]+np.pi/8, np.pi/2), num=10):
                chAz[:] = chAzi
                chEl[:] = chEli
                performanceSurf[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(performanceSurf[(model, chAzGT.r[0], chElGT.r[0])], errorFun.r)
                azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])], chAzi)
                elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])], chEli)
        chAz[:] = chAzOld
        chEl[:] = chElOld

    if key == glfw.KEY_P and action == glfw.RELEASE:
        ipdb.set_trace()
        refresh = True

    if key == glfw.KEY_S and action == glfw.RELEASE:
        print("GT Azimuth: " + str(chAzGT))
        print("Azimuth: " + str(chAz))
        print("GT Elevation: " + str(chElGT))
        print("Elevation: " + str(chEl))

        print("Dr wrt Azimuth: " + str(errorFun.dr_wrt(chAz)))
        print("Dr wrt Elevation: " + str(errorFun.dr_wrt(chEl)))
        # print("Dr wrt Distance: " + str(errorFun.dr_wrt(chDist)))


    if key == glfw.KEY_V and action == glfw.RELEASE:
        global ims
        if makeVideo:
            im_ani = animation.ArtistAnimation(figvid, ims, interval=2000, repeat_delay=3000, repeat=False, blit=True)
            im_ani.save('minimization_demo.mp4', fps=None, writer=writer, codec='mp4')
            ims = []

    if key == glfw.KEY_R and action == glfw.RELEASE:
        refresh = True

    global errorFun
    global pixelErrorFun
    global model
    if key == glfw.KEY_O and action == glfw.RELEASE:
        model = (model + 1) % 4
        if computePerformance:
            performance[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            elevations[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            azimuths[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

            performanceSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

        if model == 0:
            print("Using Gaussian model")
            errorFun = negLikModel
            pixelErrorFun = pixelLikelihoodCh
        elif model == 1:
            print("Using robust model")
            errorFun = negLikModelRobust
            pixelErrorFun = pixelLikelihoodRobustCh
        elif model == 2:
            print("Using sum of squared error model")
            errorFun = SSqE_raw
            pixelErrorFun = SE_raw
        elif model == 3:
            print("Using Gaussian Pyramid model")
            errorFun = negLikModelPyr
            pixelErrorFun = pixelLikelihoodCh

        refresh = True

    global method
    global methods
    if key == glfw.KEY_1 and action == glfw.RELEASE:
        print("Changed to minimizer: " + methods[method])
        method = 0
    if key == glfw.KEY_2 and action == glfw.RELEASE:
        print("Changed to minimizer: " + methods[method])
        method = 1
    if key == glfw.KEY_3 and action == glfw.RELEASE:
        print("Changed to minimizer: " + methods[method])
        method = 2
    if key == glfw.KEY_4 and action == glfw.RELEASE:
        print("Changed to minimizer: " + methods[method])
        method = 3
    if key == glfw.KEY_5 and action == glfw.RELEASE:
        print("Changed to minimizer: " + methods[method])
        method = 4

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
        imagegt = np.copy(np.array(rendererGT.r)).astype(np.float64)
        chImage[:,:,:] = imagegt[:,:,:]

        chAzGT[:] = chAz.r[:]
        chElGT[:] = chEl.r[:]
        chDistGT[:] = chDist.r[:]
        chComponentGT[:] = chComponent.r[:]

        changedGT = False

        if makeVideo:
            ims = []

        if computePerformance:
            figperf.clear()

        performance[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        azimuths[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        elevations[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

    if refresh:
        print("Sq Error: " + str(errorFun.r))

        edges = renderer.boundarybool_image
        imagegt = np.copy(np.array(rendererGT.r)).astype(np.float64)
        gtoverlay = imagegt.copy()
        gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
        pim1.set_data(gtoverlay)

        pim2.set_data(renderer.r)

        pim3 = ax3.imshow(pixelErrorFun.r)

        ax4.set_title("Posterior probabilities")
        ax4.imshow(np.tile(post.reshape(shapeIm[0],shapeIm[1],1), [1,1,3]))

        drazsum = pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
        img = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)

        drazsum = pixelErrorFun.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
        img = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)

        f.canvas.draw()
        plt.pause(0.1)
        refresh = False

    if minimize:
        iterat = 0
        print("Minimizing with method " + methods[method])
        ch.minimize({'raw': errorFun}, bounds=bounds, method=methods[method], x0=free_variables, callback=cb2, options={'disp':True})
        minimize = False
# ch.minimize({'raw': SSqE_pyr}, bounds=bounds, method=methods[3], x0=free_variables, callback=cb2, options={'disp':True})

# elapsed_time = time.time() - mintime
# print("Minimization time:  " + str(elapsed_time))

edges = renderer.boundarybool_image
gtoverlay = imagegt.copy()
gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
pim1.set_data(gtoverlay)

pim2.set_data(renderer.r)

pim3 = ax3.imshow(pixelErrorFun.r)

ax4.set_title("Posterior probabilities")
ax4.imshow(np.tile(post.reshape(shapeIm[0],shapeIm[1],1), [1,1,3]))

ax5.set_title("Dr wrt. Azimuth")
drazsum = pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)

img = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)

ax6.set_title("Dr wrt. Elevation")
drazsum = pixelErrorFun.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)

img = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)

f.canvas.draw()
plt.pause(0.1)

plt.imsave('opendr_opengl_final.png', rendererGT.r)
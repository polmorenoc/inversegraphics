import saveExrImages
reload(saveExrImages)
from saveExrImages import exportExrImages
import os
print "Reading xml "
for sceneRoot, sceneDirs, files in os.walk("../output/annotation/"):
    # for sceneName in sceneDirs:
    #     sceneDir = os.path.join(sceneRoot, sceneName)
    #     for modelRoot, modelDirs, files2 in os.walk(sceneDir):
    #         for modelName in modelDirs:
    #             modelDir = os.path.join(modelRoot, modelName)
    #             numModels = 1
    for xmlScene in files:
        if xmlScene.endswith(".exr"): 

            if not xmlScene.endswith("_single.exr"): 
                print "Reading xml " + xmlScene
                try:
                    outfilename = os.path.splitext(xmlScene)[0]

                    print "Reading " + outfilename

                    #sceneFullPath = os.path.splitext(modelDir + '/' + xmlScene)[0]

                    exportExrImages("../output/annotation/", "../output/images/", outfilename)
                except Exception as e:
                    print e
                    print "Scene " + outfilename + " failed!"
    #            numModels = numModels + 1
                        

import save_exr_images
from save_exr_images import exportExrImages
import os
print ("Reading xml ")


outputDir = '../data/output/'
imgDir = outputDir  + "images/"

lines = [line.strip() for line in open(outputDir  + 'groundtruth.txt')]

if not os.path.exists(imgDir):
    os.makedirs(imgDir)

for instance in lines:

    parts = instance.split(' ')
    teapot = int(parts[3])
    frame = int(parts[4])

    sceneNum = int(parts[5])

    targetIndex = int(parts[6])

    prefix = ''
    if len(parts) == 17:
        prefix = parts[16]
    try:
        exportExrImages(outputDir, imgDir, teapot, frame, sceneNum, targetIndex, prefix)
    except Exception as e:
        print(e)


#!/usr/bin/python

import OpenEXR
import Imath
from PIL import Image
import sys
import numpy as np

def exportExrImages(annotationdir, imgdir, numTeapot, frame, sceneNum, target, prefix):

    framestr = '{0:04d}'.format(frame)

    outfilename = "render" + prefix + "_obj" + str(numTeapot) + "_scene" + str(sceneNum) + "_target" + str(target)  + "_" + framestr
    exrfile = OpenEXR.InputFile(annotationdir + outfilename + ".exr")
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = exrfile.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    quality_val = 100

    # rgbf = [Image.fromstring("F", size, exrfile.channel("shNormal." + c, pt)) for c in "RGB"]
    # extrema = [im.getextrema() for im in rgbf]
    # darkest = min([lo for (lo,hi) in extrema])
    # lighest = max([hi for (lo,hi) in extrema])
    # scale = 255 / (lighest - darkest)
    # def normalize_0_255(v):
    #     return (v * scale) + darkest
    # rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]
    # Image.merge("RGB", rgb8).save(jpgfilename + "_normal.jpg", "JPEG")

    rgbf = [Image.fromstring("F", size, exrfile.channel("RenderLayer.Combined." + c, pt)) for c in "RGB"]
    pix = [np.array(im) for im in rgbf]
    pix[0][pix[0]>1] = 1
    pix[1][pix[1]>1] = 1
    pix[2][pix[2]>1] = 1
    pix[0] = pix[0]*255
    pix[1] = pix[1]*255
    pix[2] = pix[2]*255
    imgr = Image.fromarray(pix[0].astype('uint8'))
    imgg = Image.fromarray(pix[1].astype('uint8'))
    imgb = Image.fromarray(pix[2].astype('uint8'))
    finalimg = Image.merge("RGB", (imgr, imgg, imgb))
    # finalimg.save(imgdir + outfilename + ".png", "PNG", quality=quality_val)

    distancestr = exrfile.channel('RenderLayer.IndexOB.X', pt)
    distance = Image.fromstring("F", size, distancestr)


    shapeIndexstr = exrfile.channel('RenderLayer.IndexOB.X', pt)
    shapeIndex = Image.fromstring("F", size, shapeIndexstr)
    segment = np.array(shapeIndex)
    sumComplete = np.sum(segment)
    segmentimg = Image.fromarray(segment.astype('uint8')*255)
    segmentimg.save(imgdir + outfilename + "_segment.png", "PNG", quality=quality_val)


    singlefile = "render" + prefix + "_obj" + str(numTeapot) + "_scene" + str(sceneNum) + "_target" + str(target)  + "_single_" + framestr
    exrfile = OpenEXR.InputFile(annotationdir + singlefile + ".exr")

    rgbf = [Image.fromstring("F", size, exrfile.channel("RenderLayer.001.Combined." + c, pt)) for c in "RGB"]
    pix = [np.array(im) for im in rgbf]
    pix[0][pix[0]>1] = 1
    pix[1][pix[1]>1] = 1
    pix[2][pix[2]>1] = 1
    pix[0] = pix[0]*255
    pix[1] = pix[1]*255
    pix[2] = pix[2]*255
    imgr = Image.fromarray(pix[0].astype('uint8'))
    imgg = Image.fromarray(pix[1].astype('uint8'))
    imgb = Image.fromarray(pix[2].astype('uint8'))
    finalimg = Image.merge("RGB", (imgr, imgg, imgb))
    finalimg.save(imgdir + singlefile + ".png", "PNG", quality=quality_val)

    shapeIndexstrsingle = exrfile.channel('RenderLayer.001.IndexOB.X', pt)
    shapeIndexsingle = Image.fromstring("F", size, shapeIndexstrsingle)
    segmentsingle = np.array(shapeIndexsingle)

    segmentimgsingle = Image.fromarray(segmentsingle.astype('uint8')*255)
    segmentimgsingle.save(imgdir + singlefile + "_segment.png", "PNG", quality=quality_val)
    sumSingle = np.sum(segmentsingle)

    print "Sum Complete " + str(sumComplete)
    print "Sum Single " + str(sumSingle)
    with open(annotationdir + 'occlusions' + ".txt", "a") as myfile:
        myfile.write(str(numTeapot) + ' ' + str(frame) + ' ' + str(sceneNum) + " " + str(target) + " " + str(sumComplete/sumSingle) + ' ' + prefix + "\n")

    return

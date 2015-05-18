#!/usr/bin/python

import OpenEXR
import Imath
from PIL import Image
import sys
import numpy as np

def exportExrImages(annotationdir, imgdir, numTeapot, frame):

    framestr = '{0:04d}'.format(frame)
    outfilename = "scene_obj" + str(numTeapot) + "_" + framestr
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
    finalimg.save(imgdir + outfilename + ".png", "PNG", quality=quality_val)


    distancestr = exrfile.channel('RenderLayer.IndexOB.X', pt)
    distance = Image.fromstring("F", size, distancestr)


    # rgbf = Image.fromstring("F", size, exrfile.channel("RenderLayer.IndexOB.X", pt))
    # pix = np.array(rgbf)
    # # idx = pix[:,:] > 200.0
    # # pix[idx] = 200.0
    # pix = abs(np.max(pix) - pix)
    # scale = 255 / (np.max(pix) - np.min(pix))
    # pix = pix*scale + np.min(pix)

    # distanceimg = Image.fromarray(pix.astype('uint8'))
    # finaldistanceimg = Image.merge("RGB", (distanceimg, distanceimg, distanceimg))
    # finaldistanceimg.save(jpgfilename + "_distance.jpg", "JPEG")

   
    shapeIndexstr = exrfile.channel('RenderLayer.IndexOB.X', pt)
    shapeIndex = Image.fromstring("F", size, shapeIndexstr)
    segment = np.array(shapeIndex)
    segment[segment > 1.1] = 1
    segment[segment < 0.90] = 0
    sumComplete = np.sum(segment)
    segmentimg = Image.fromarray(segment.astype('uint8')*255)


    singlefile = "scene_obj" + str(numTeapot) + "_single" + framestr

    exrfile = OpenEXR.InputFile(annotationdir + singlefile + ".exr")

    shapeIndexstrsingle = exrfile.channel('RenderLayer.001.IndexOB.X', pt)
    shapeIndexsingle = Image.fromstring("F", size, shapeIndexstrsingle)
    segmentsingle = np.array(shapeIndexsingle)
    segmentsingle[segmentsingle > 1.1] = 1
    segmentsingle[segmentsingle < 0.95] = 0
    segmentimgsingle = Image.fromarray(segmentsingle.astype('uint8')*255)
    # segmentimgsingle.convert('RGB').save(annotationdir + outfilename + "_single_primIndex.jpg", "JPEG", quality=quality_val)
    sumSingle = np.sum(segmentsingle)

    print "Sum Complete " + str(sumComplete)
    print "Sum Single " + str(sumSingle)
    with open(annotationdir + 'occlusions' + ".txt", "a") as myfile:
        myfile.write(str(numTeapot) + ' ' + str(frame) + ' ' + str(sumComplete/sumSingle) + "\n")

    return
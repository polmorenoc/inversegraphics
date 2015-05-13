#!/usr/bin/python

import OpenEXR
import Imath
from PIL import Image
import sys
import numpy as np

def exportExrImages(annotationdir, imgdir, outfilename):
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

    rgbf = [Image.fromstring("F", size, exrfile.channel("RenderLayer.001.Combined." + c, pt)) for c in "RGB"]
    extrema = [im.getextrema() for im in rgbf]
    darkest = min([lo for (lo,hi) in extrema])
    lighest = max([hi for (lo,hi) in extrema])
    scale = 255 / (1.0 - darkest)
    def normalize_0_255(v):
        return (v * scale) + darkest
    rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]
    Image.merge("RGB", rgb8).show()
    # save(imgdir + outfilename + ".jpg", "JPEG", quality=quality_val)


    distancestr = exrfile.channel('distance.Y', pt)
    distance = Image.fromstring("F", size, distancestr)
    distance.show()

    rgbf = Image.fromstring("F", size, exrfile.channel("distance.Y", pt))
    # def cutVal(v):
    #     if v > 1000:
    #         return 1000
    #     else:
    #         return v
    # rgbfcut = rgbf.point(cutVal, "1")
    
    # extrema = rgbfcut.getextrema()
    # darkest = min(extrema)
    # lighest = max(extrema)
    # scale = 255 / (lighest - darkest)
    # def normalize_0_255(v):
    #     return (v * scale) + darkest
    # rgb8 = rgbfcut.point(normalize_0_255,"1")
    # rgb8.save(jpgfilename + "_distance.jpg", "JPEG")
    pix = np.array(rgbf)
    idx = pix[:,:] > 200.0
    pix[idx] = 200.0
    pix = abs(np.max(pix) - pix)
    scale = 255 / (np.max(pix) - np.min(pix))
    pix = pix*scale + np.min(pix)

    distanceimg = Image.fromarray(pix.astype('uint8'))
    finaldistanceimg = Image.merge("RGB", (distanceimg, distanceimg, distanceimg))
    finaldistanceimg.save(jpgfilename + "_distance.jpg", "JPEG")

   

    shapeIndexstr = exrfile.channel('RenderLayer.IndexOB.X', pt)
    shapeIndex = Image.fromstring("F", size, shapeIndexstr)
    segment = np.array(shapeIndex)
    segment[segment > 1.1] = 0;
    segment[segment < 0.95] = 0;
    sumComplete = np.sum(segment)
    segmentimg = Image.fromarray(segment.astype('uint8')*255)
    # segmentimg.convert('RGB').save(annotationdir + outfilename + "_primIndex.jpg", "JPEG", quality=quality_val)
    # rgbf = [shapeIndex, shapeIndex, shapeIndex]
    # extrema = [im.getextrema() for im in rgbf]
    # darkest = min([lo for (lo,hi) in extrema])
    # lighest = max([hi for (lo,hi) in extrema])
    # scale = 255 / (lighest - darkest)
    # def normalize_0_255(v):
    #     return (v * scale) + darkest
    # rgb8 = [im.convert("L") for im in rgbf]
    # Image.merge("RGB", rgb8).save(jpgfilename + "_primIndex.jpg", "JPEG")


    exrfile = OpenEXR.InputFile(annotationdir + outfilename + "_single.exr")
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = exrfile.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    shapeIndexstrsingle = exrfile.channel('primIndex.Y', pt)
    shapeIndexsingle = Image.fromstring("F", size, shapeIndexstrsingle)
    segmentsingle = np.array(shapeIndexsingle)
    segmentsingle[segmentsingle > 1.1] = 0;
    segmentsingle[segmentsingle < 0.95] = 0;
    segmentimgsingle = Image.fromarray(segmentsingle.astype('uint8')*255)
    # segmentimgsingle.convert('RGB').save(annotationdir + outfilename + "_single_primIndex.jpg", "JPEG", quality=quality_val)
    sumSingle = np.sum(segmentsingle)

    print "Sum Complete " + str(sumComplete)
    print "Sum Single " + str(sumSingle)
    with open(annotationdir + outfilename + ".txt", "a") as myfile:
        myfile.write('\n' + str(sumComplete/sumSingle))

    return;

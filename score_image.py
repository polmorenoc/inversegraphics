import cv2
import numpy
import matplotlib.pyplot as plt


def getChamferDistance(img, template, minThresImage, maxThresImage, minThresTemplate, maxThresTemplate):
    imgEdges = cv2.Canny(numpy.uint8(img*255), minThresImage,maxThresImage)
    tempEdges = cv2.Canny(numpy.uint8(template*255), minThresTemplate, maxThresTemplate)

    bwEdges1 = cv2.distanceTransform(~tempEdges, cv2.DIST_L2, 5)

    score = numpy.sum(numpy.multiply(imgEdges/255, bwEdges1))

    return score


def testImageMatching():
    teapots = ["test/teapot1", "test/teapot2","test/teapot3","test/teapot4","test/teapot5","test/teapot6"]

    images = []
    edges = []
    for teapot in teapots:
        im = cv2.imread(teapot + ".png")
        can = cv2.Canny(im, 50,255)
        images.append(im)
        edges.append(can)
        cv2.imwrite(teapot + "_can.png", can)

    confusion = numpy.zeros([6,6])
    for tp1 in numpy.arange(1,7):
        for tp2 in numpy.arange(tp1,7):
            dist = getChamferDistance(images[tp1-1], images[tp2-1])
            print(dist)
            confusion[tp1-1, tp2-1] = dist

    plt.imshow(confusion, interpolation='nearest')
    plt.savefig('test/confusion.png')



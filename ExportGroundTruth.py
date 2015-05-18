import saveExrImages
reload(saveExrImages)
from saveExrImages import exportExrImages
import os
print "Reading xml "

lines = [line.strip() for line in open('output/groundtruth.txt')]

for instance in lines:
    parts = instance.split(' ')
    teapot = int(parts[3])
    frame = int(parts[4])
    prefix = ''
    if len(parts) == 6:
        prefix = parts[5]

    try:
        exportExrImages("output/", "output/images/", teapot, frame, prefix)
    except Exception as e:
        print e


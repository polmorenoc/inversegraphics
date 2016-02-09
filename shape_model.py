import numpy as np
import pickle
import chumpy as ch
import ipdb

#%% Helper functions
def longToPoints3D(pointsLong):
    nPointsLong = np.size(pointsLong)
    return np.reshape(pointsLong, [nPointsLong/3, 3])


def shapeParamsToVerts(shapeParams, teapotModel):
    landmarksLong = shapeParams.dot(teapotModel['ppcaW'].T) + teapotModel['ppcaB']
    landmarks = longToPoints3D(landmarksLong)
    vertices = teapotModel['meshLinearTransform'].dot(landmarks)
    return vertices

def chShapeParamsToVerts(shapeParams, meshLinearTransform, ppcaW, ppcaB):
    landmarksLong = ch.dot(shapeParams,ppcaW.T) + ppcaB
    landmarks = landmarksLong.reshape([-1,3])
    vertices = ch.dot(meshLinearTransform,landmarks)
    return vertices

def saveObj(vertices, faces, normals, filePath):
    with open(filePath, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        for n in normals:
            f.write("vn %.4f %.4f %.4f\n" % (n[0], n[1], n[2]))
        for p in faces:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")

def loadObject(fileName):
    with open(fileName, 'rb') as inpt:
        return pickle.load(inpt)

def normalize_v3(arr):
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr


def getNormals(vertices, faces):
    norm = np.zeros( vertices.shape, dtype=vertices.dtype )
    tris = vertices[faces]
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    normalize_v3(n)
    norm[ faces[:,0] ] += n
    norm[ faces[:,1] ] += n
    norm[ faces[:,2] ] += n
    normalize_v3(norm)
    return norm

def chGetNormals(vertices, faces):
    import opendr.geometry
    return opendr.geometry.VertNormals(vertices, faces).reshape((-1,3))



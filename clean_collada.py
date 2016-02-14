import pyassimp
from pyassimp.postprocess import *

modelPath = '../databaseFull/models/teapots/78ed1a0383cd2eab7552b099aebcb24e/Teapot_fixed.dae'

# aiProcessPreset_TargetRealtime_Quality = ( \
#     aiProcess_CalcTangentSpace              |  \
#     aiProcess_GenSmoothNormals              |  \
#     aiProcess_JoinIdenticalVertices         |  \
#     aiProcess_ImproveCacheLocality          |  \
#     aiProcess_LimitBoneWeights              |  \
#     aiProcess_RemoveRedundantMaterials      |  \
#     aiProcess_SplitLargeMeshes              |  \
#     aiProcess_Triangulate                   |  \
#     aiProcess_GenUVCoords                   |  \
#     aiProcess_SortByPType                   |  \
#     aiProcess_FindDegenerates               |  \
#     aiProcess_FindInvalidData               |  \
#     0 )

postprocess = aiProcessPreset_TargetRealtime_Quality

scene = pyassimp.load(modelPath, postprocess)

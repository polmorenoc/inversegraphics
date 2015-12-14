import ipdb
import PyTorchAug
import PyTorch
nn = PyTorch.Nn()

lua = PyTorchAug.lua
lua.getGlobal("require")
lua.pushString('modules/LinearCR')
lua.call(1, 0)

lua = PyTorchAug.lua
lua.getGlobal("require")
lua.pushString('modules/Reparametrize')
lua.call(1, 0)

lua = PyTorchAug.lua
lua.getGlobal("require")
lua.pushString('modules/SelectiveOutputClamp')
lua.call(1, 0)

lua = PyTorchAug.lua
lua.getGlobal("require")
lua.pushString('modules/SelectiveGradientFilter')
lua.call(1, 0)


dim_hidden = 200
feature_maps = 96

filter_size = 5
colorchaPyTorchAugels = 1

# ipdb.set_trace()

encoder = PyTorchAug.Sequential()


encoder.add(PyTorchAug.SpatialConvolution(colorchaPyTorchAugels,feature_maps,filter_size,filter_size))

encoder.add(PyTorchAug.SpatialMaxPooling(2,2,2,2))
encoder.add(PyTorchAug.Threshold(0,1e-6))

encoder.add(PyTorchAug.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size))
encoder.add(PyTorchAug.SpatialMaxPooling(2,2,2,2))
encoder.add(PyTorchAug.Threshold(0,1e-6))



encoder.add(PyTorchAug.SpatialConvolution(feature_maps/2,feature_maps/4,filter_size,filter_size))
encoder.add(PyTorchAug.SpatialMaxPooling(2,2,2,2))
encoder.add(PyTorchAug.Threshold(0,1e-6))

encoder.add(PyTorchAug.Reshape((feature_maps/4)*15*15))

z = PyTorchAug.ConcatTable()

mu = PyTorchAug.Sequential()

mu.add(PyTorchAug.LinearCR((feature_maps/4)*15*15, dim_hidden))
mu.add(PyTorchAug.SelectiveGradientFilter())
mu.add(PyTorchAug.SelectiveOutputClamp())
z.add(mu)

sigma = PyTorchAug.Sequential()
sigma.add(PyTorchAug.LinearCR((feature_maps/4)*15*15, dim_hidden))
sigma.add(PyTorchAug.SelectiveGradientFilter())
sigma.add(PyTorchAug.SelectiveOutputClamp())
z.add(sigma)


encoder.add(z)

decoder = PyTorchAug.Sequential()
decoder.add(PyTorchAug.LinearCR(dim_hidden, (feature_maps/4)*15*15 ))
decoder.add(PyTorchAug.Threshold(0,1e-6))

decoder.add(PyTorchAug.Reshape((feature_maps/4),15,15))

decoder.add(PyTorchAug.SpatialUpSamplingNearest(2))
decoder.add(PyTorchAug.SpatialConvolution(feature_maps/4,feature_maps/2, 7, 7))
decoder.add(PyTorchAug.Threshold(0,1e-6))

decoder.add(PyTorchAug.SpatialUpSamplingNearest(2))
decoder.add(PyTorchAug.SpatialConvolution(feature_maps/2,feature_maps,7,7))
decoder.add(PyTorchAug.Threshold(0,1e-6))

decoder.add(PyTorchAug.SpatialUpSamplingNearest(2))
decoder.add(PyTorchAug.SpatialConvolution(feature_maps,feature_maps,7,7))
decoder.add(PyTorchAug.Threshold(0,1e-6))

decoder.add(PyTorchAug.SpatialUpSamplingNearest(2))
decoder.add(PyTorchAug.SpatialConvolution(feature_maps,1,7,7))
decoder.add(PyTorchAug.Sigmoid())

model = PyTorchAug.Sequential()
model.add(encoder)
model.add(PyTorchAug.Reparametrize(dim_hidden))
model.add(decoder)

model.cuda()
nn.collectgarbage()
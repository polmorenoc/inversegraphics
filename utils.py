import numpy as np
import os

__author__ = 'pol'

def generateExperiment(size, prefix, ratio, seed):
    np.random.seed(seed)
    data = np.arange(size)
    np.random.shuffle(data)
    train = data[0:np.int(size*ratio)]
    test = data[np.int(size*ratio)::]

    if not os.path.exists('experiment/' + prefix):
        os.makedirs('experiment/' + prefix)

    np.save('experiment/' + prefix + 'train.npy', train)
    np.save('experiment/' + prefix + 'test.npy', test)
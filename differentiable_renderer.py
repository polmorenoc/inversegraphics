import chumpy as ch
from chumpy import depends_on, Ch
import cv2
import numpy as np
import scipy.sparse as sp
from chumpy.utils import row, col
from opendr.geometry import Rodrigues

#Make simple experiment.

class DifferentiableRenderer(Ch):
    terms = ['renderer', 'camera']
    dterms = ['params']

    def compute_r(self):
        return self.renderer.r

    def compute_dr_wrt(self, wrt):
        for param in params:
            if wrt is param:
                #Perform occlusion differentiation.
        else:
            return renderer.dr_wrt(wrt)

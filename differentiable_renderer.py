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
        for param in self.params:
            if wrt is param:
                jac = self.renderer.dr_wrt(wrt)
                camJac = self.camera.dr_wrt(param)
                boundary = self.renderer.boundaryid_image
                bnd_bool = boundary != 4294967295
                barycentric = self.renderer.barycentric_image
                visibility = self.renderer.visibility_image
                lidxs_out, ridxs_out, tidxs_out, bidxs_out, lidxs_int, ridxs_int, tidxs_int, bidxs_int = self.renderer.boundary_neighborhood(bnd_bool)




        else:
            return self.renderer.dr_wrt(wrt)

import chumpy as ch
from chumpy import depends_on, Ch
import cv2
import numpy as np
import scipy.sparse as sp
from chumpy.utils import row, col
from opendr.geometry import Rodrigues


class RotateZ(Ch):
    dterms = 'a'

    def compute_r(self):
        return np.array([[np.cos(self.a.r), -np.sin(self.a.r), 0, 0], [np.sin(self.a.r), np.cos(self.a.r), 0, 0], [0, 0, 1, 0], [0,0,0,1]])
        
    def compute_dr_wrt(self, wrt):

        if wrt is not self.a:
            return
        
        if wrt is self.a:
            return np.array([[-np.sin(self.a.r)[0], -np.cos(self.a.r)[0], 0, 0], [np.cos(self.a.r)[0], -np.sin(self.a.r)[0], 0, 0], [0, 0, 0, 0], [0,0,0,0]]).reshape(16,1)


class RotateX(Ch):
    dterms = 'a'

    def compute_r(self):
        return np.array([[1, 0, 0, 0],[0, np.cos(self.a.r), -np.sin(self.a.r), 0], [0, np.sin(self.a.r), np.cos(self.a.r),0],[0,0,0,1]])

    def compute_dr_wrt(self, wrt):

        # if wrt is not self.a:
        #     return

        if wrt is self.a:
            return np.array([[0, 0, 0, 0],[0, -np.sin(self.a.r)[0], -np.cos(self.a.r)[0], 0], [0, np.cos(self.a.r)[0], -np.sin(self.a.r)[0],0],[0,0,0,0]]).reshape(16,1)

class Translate(Ch):
    dterms = 'x', 'y', 'z'

    def compute_r(self):
        return np.array([[1, 0, 0, self.x.r],[0, 1, 0, self.y.r], [0, 0, 1,self.z.r],[0,0,0,1]])

    def compute_dr_wrt(self, wrt):

        if wrt is not self.x and wrt is not self.y and wrt is not self.z:
            return

        if wrt is self.x:
            return np.array([[0, 0, 0, 1],[0, 0, 0, 0], [0, 0, 0,0],[0,0,0,0]]).reshape(16,1)
        if wrt is self.y:
            return np.array([[0, 0, 0, 0],[0, 0, 0, 1], [0, 0, 0,0],[0,0,0,0]]).reshape(16,1)
        if wrt is self.z:
            return np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 1],[0,0,0,0]]).reshape(16,1)
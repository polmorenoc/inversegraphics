import chumpy as ch
from chumpy import depends_on, Ch
import cv2
import numpy as np
import scipy.sparse as sp
from chumpy.utils import row, col
from opendr.geometry import Rodrigues
import warnings

#Make simple experiment.
def nanmean(a, axis):
    # don't call nan_to_num in here, unless you check that
    # occlusion_test.py still works after you do it!
    result = np.nanmean(a, axis=axis)
    return result

def nangradients(arr):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dy = np.expand_dims(arr[:-1,:,:] - arr[1:,:,:], axis=3)
        dx = np.expand_dims(arr[:,:-1,:] - arr[:, 1:, :], axis=3)

        dy = np.concatenate((dy[1:,:,:], dy[:-1,:,:]), axis=3)
        dy = np.nanmean(dy, axis=3)
        dx = np.concatenate((dx[:,1:,:], dx[:,:-1,:]), axis=3)
        dx = np.nanmean(dx, axis=3)

        if arr.shape[2] > 1:
            gy, gx, _ = np.gradient(arr)
        else:
            gy, gx = np.gradient(arr.squeeze())
            gy = np.atleast_3d(gy)
            gx = np.atleast_3d(gx)
        gy[1:-1,:,:] = -dy
        gx[:,1:-1,:] = -dx

    return gy, gx


class SQErrorRenderer(Ch):
    terms = ['renderer', 'params_list']
    dterms = ['params']

    def compute_r(self):
        return self.renderer.errors

    def compute_dr_wrt(self, wrt):
        import ipdb
        # ipdb.set_trace()
        inParamsList = False
        if wrt is self.params:
            return None

        if not inParamsList:
            for param in self.params_list:
                if wrt is param:
                    inParamsList = True

        if inParamsList:
            return self.gradient_pred(wrt)
        else:
            errJac = sp.csc_matrix((2*self.renderer.r - 2*self.imageGT).ravel()[:,None])
            return errJac.multiply(self.renderer.dr_wrt(wrt))

    def gradient_pred(self, paramWrt):

        observed = self.renderer.color_image
        boundaryid_image = self.renderer.boundaryid_image
        barycentric = self.renderer.barycentric_image
        visibility = self.renderer.visibility_image
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        jacIm = self.dImage_wrt_2dVerts_predict(observed, paramWrt, visible, visibility, barycentric, observed.shape[0], observed.shape[1], self.renderer.v.shape[0], self.renderer.f, boundaryid_image != 4294967295)
        return jacIm.dot(self.renderer.camera.dr_wrt(paramWrt))

    def dImage_wrt_2dVerts_predict(self, observed, paramWrt, visible, visibility, barycentric, image_width, image_height, num_verts, f, bnd_bool):
        """Construct a sparse jacobian that relates 2D projected vertex positions
        (in the columns) to pixel values (in the rows). This can be done
        in two steps."""
        bnd_bool = np.logical_and(bnd_bool, self.renderer.visibility_image != 4294967295)

        camJac = self.renderer.camera.dr_wrt(paramWrt)

        xdiff = self.renderer.dEdx
        ydiff = self.renderer.dEdy

        # visible.ravel()[bidxs_out[bidx]] = True

        # import ipdb
        # ipdb.set_trace()

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]

        n_channels = np.atleast_3d(observed).shape[2]
        shape = visibility.shape

        #2: Take the data and copy the corresponding dxs and dys to these new pixels.

        # Step 1: get the structure ready, ie the IS and the JS
        IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
        JS = col(f[visibility.ravel()[visible]].ravel())
        JS = np.hstack((JS*2, JS*2+1)).ravel()

        pxs = np.asarray(visible % shape[1], np.int32)
        pys = np.asarray(np.floor(np.floor(visible) / shape[1]), np.int32)

        if n_channels > 1:
            IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
            JS = np.concatenate([JS for i in range(n_channels)])

        datas = []

        # The data is weighted according to barycentric coordinates
        bc0 = col(barycentric[pys, pxs, 0])
        bc1 = col(barycentric[pys, pxs, 1])
        bc2 = col(barycentric[pys, pxs, 2])
        for k in range(n_channels):
            dxs = xdiff[pys, pxs, k]
            dys = ydiff[pys, pxs, k]
            if f.shape[1] == 3:
                datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1,col(dxs)*bc2,col(dys)*bc2)).ravel())
            else:
                datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1)).ravel())

        data = np.concatenate(datas)

        ij = np.vstack((IS.ravel(), JS.ravel()))

        result = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

        return result


class DifferentiableRenderer(Ch):
    terms = ['renderer', 'params_list']
    dterms = ['params']

    def compute_r(self):
        return self.renderer.r

    def compute_dr_wrt(self, wrt):
        import ipdb

        for param in self.params_list:
            if wrt is param:
                return self.gradient_pred(wrt)

        return self.renderer.dr_wrt(wrt)


    def gradients(self):
        # self._call_on_changed()

        observed = self.renderer.color_image
        boundaryid_image = self.renderer.boundaryid_image
        barycentric = self.renderer.barycentric_image
        visibility = self.renderer.visibility_image
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        return self.dImage_wrt_2dVerts_bnd_gradient(observed, barycentric, observed.shape[0], observed.shape[1], boundaryid_image != 4294967295)

    def dImage_wrt_2dVerts_bnd_gradient(self, observed, barycentric, image_width, image_height, bnd_bool):

        bnd_bool = np.logical_and(self.renderer.visibility_image != 4294967295, bnd_bool)

        n_channels = np.atleast_3d(observed).shape[2]
        shape = [image_height, image_width]

        bndf = bnd_bool.astype(np.float64)

        bnd_nan = bndf.reshape((observed.shape[0], observed.shape[1], -1)).copy()
        bnd_nan.ravel()[bnd_nan.ravel()>0] = np.nan
        bnd_nan += 1
        obs_nonbnd = np.atleast_3d(observed) * bnd_nan

        ydiffnb, xdiffnb = nangradients(obs_nonbnd)

        observed = np.atleast_3d(observed)

        if observed.shape[2] > 1:
            ydiffbnd, xdiffbnd, _ = np.gradient(observed)
        else:
            ydiffbnd, xdiffbnd = np.gradient(observed.squeeze())
            ydiffbnd = np.atleast_3d(ydiffbnd)
            xdiffbnd = np.atleast_3d(xdiffbnd)

        # This corrects for a bias imposed boundary differences begin spread over two pixels
        # (by np.gradients or similar) but only counted once (since OpenGL's line
        # drawing spans 1 pixel)
        xdiffbnd *= 2.0
        ydiffbnd *= 2.0

        xdiffnb = -xdiffnb
        ydiffnb = -ydiffnb
        xdiffbnd = -xdiffbnd
        ydiffbnd = -ydiffbnd

        idxs = np.isnan(xdiffnb.ravel())
        xdiffnb.ravel()[idxs] = xdiffbnd.ravel()[idxs]

        idxs = np.isnan(ydiffnb.ravel())
        ydiffnb.ravel()[idxs] = ydiffbnd.ravel()[idxs]

        xdiff = xdiffnb
        ydiff = ydiffnb

        dybt = -np.vstack([np.diff(observed, n=1, axis=0), np.zeros([1,observed.shape[1],3])])
        dytb = np.vstack([np.zeros([1,observed.shape[1],3]), np.flipud(np.diff(np.flipud(observed), n=1, axis=0))])

        dxrl = -np.hstack([np.diff(observed, n=1, axis=1), np.zeros([observed.shape[0],1,3])])
        dxlr = np.hstack([np.zeros([observed.shape[0],1,3]),np.fliplr(np.diff(np.fliplr(observed), n=1, axis=1))])

        bary_sl = np.roll(barycentric , shift=-1, axis=1)
        bary_sr = np.roll(barycentric , shift=1, axis=1)
        bary_st = np.roll(barycentric , shift=-1, axis=0)
        bary_sb = np.roll(barycentric , shift=1, axis=0)

        return xdiff, ydiff, dybt, dxrl, dytb, dxlr, bary_sl, bary_sr, bary_st, bary_sb

    def gradient_pred(self, paramWrt):
        observed = self.renderer.color_image
        boundaryid_image = self.renderer.boundaryid_image
        barycentric = self.renderer.barycentric_image
        visibility = self.renderer.visibility_image
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        jacIm = self.dImage_wrt_2dVerts_predict(observed, paramWrt, visible, visibility, barycentric, observed.shape[0], observed.shape[1], self.renderer.v.shape[0], self.renderer.f, boundaryid_image != 4294967295)
        return jacIm.dot(self.renderer.camera.dr_wrt(paramWrt))

    def boundary_neighborhood(self):
        boundary = self.renderer.boundaryid_image != 4294967295
        visibility = self.renderer.visibility_image != 4294967295

        boundary = np.logical_and(visibility, boundary)
        shape = boundary.shape

        notboundary = np.logical_not(boundary)
        horizontal = np.hstack((np.diff(boundary.astype(np.int8),axis=1), np.zeros((shape[0],1), dtype=np.int8)))
        # horizontal = np.hstack((np.diff(boundary.astype(np.int8),axis=1), np.zeros((shape[0],1), dtype=np.int8)))
        vertical = np.vstack((np.diff(boundary.astype(np.int8), axis=0), np.zeros((1,shape[1]), dtype=np.int8)))
        # vertical = np.vstack((np.diff(boundary.astype(np.int8), axis=0), np.zeros((1,shape[1]), dtype=np.int8)))

        pixl = (horizontal == 1)
        pixr = (horizontal == -1)
        pixt = (vertical == 1)
        pixb = (vertical == -1)

        # plt.imshow((pixrl | pixlr | pixtb | pixbt))

        #Quicker, convolve (FFT) and take mask * etc.

        lidxs_out = np.where(pixl.ravel())[0]
        ridxs_out = np.where(pixr.ravel())[0] + 1
        tidxs_out = np.where(pixt.ravel())[0]
        bidxs_out = np.where(pixb.ravel())[0] + shape[1]
        lidxs_int = np.where(pixl.ravel())[0] + 1
        ridxs_int = np.where(pixr.ravel())[0]
        tidxs_int = np.where(pixt.ravel())[0] + shape[1]
        bidxs_int = np.where(pixb.ravel())[0]


        return pixr, pixl, pixt, pixb, lidxs_out, ridxs_out, tidxs_out, bidxs_out, lidxs_int, ridxs_int, tidxs_int, bidxs_int

    def dImage_wrt_2dVerts_predict(self, observed, paramWrt, visible, visibility, barycentric, image_width, image_height, num_verts, f, bnd_bool):
        """Construct a sparse jacobian that relates 2D projected vertex positions
        (in the columns) to pixel values (in the rows). This can be done
        in two steps."""
        bnd_bool = np.logical_and(bnd_bool, self.renderer.visibility_image != 4294967295)

        camJac = self.renderer.camera.dr_wrt(paramWrt)

        xdiff, ydiff, dybt, dxrl, dytb, dxlr, bary_sl, bary_sr, bary_st, bary_sb = self.gradients()

        pixr, pixl, pixt, pixb, lidxs_out, ridxs_out, tidxs_out, bidxs_out, lidxs_int, ridxs_int, tidxs_int, bidxs_int = self.boundary_neighborhood()

        lidxs_out = np.where(bnd_bool.ravel())[0]-1
        ridxs_out = np.where(bnd_bool.ravel())[0]+1
        tidxs_out = np.where(bnd_bool.ravel())[0]-bnd_bool.shape[1]
        bidxs_out = np.where(bnd_bool.ravel())[0]+bnd_bool.shape[1]

        lidxs_int = np.where(bnd_bool.ravel())[0]
        ridxs_int= np.where(bnd_bool.ravel())[0]
        tidxs_int= np.where(bnd_bool.ravel())[0]
        bidxs_int = np.where(bnd_bool.ravel())[0]

        #Where are triangles moving wrt to the image coordinates at the boundaries?
        lintGrad = camJac[f[visibility.ravel()[lidxs_int]]*2]
        rintGrad = camJac[f[visibility.ravel()[ridxs_int]]*2]
        tintGrad = camJac[f[visibility.ravel()[tidxs_int]]*2+1]
        bintGrad = camJac[f[visibility.ravel()[bidxs_int]]*2+1]

        lidx = lintGrad[:,0,0] < -0.0001
        xdiff.reshape([-1,3])[lidxs_out[lidx]] = xdiff.reshape([-1,3])[lidxs_int[lidx]]
        barycentric.reshape([-1,3])[lidxs_out[lidx]] = barycentric.reshape([-1,3])[lidxs_int[lidx]]
        visibility.ravel()[lidxs_out[lidx]] = visibility.ravel()[lidxs_int[lidx]]
        xdiff.reshape([-1,3])[lidxs_int[lidx]] = dxrl.reshape([-1,3])[lidxs_int[lidx]]

        # visible.ravel()[lidxs_out[lidx]] = True

        ridx = rintGrad[:,0,0] > 0.0001
        xdiff.reshape([-1,3])[ridxs_out[ridx]] = xdiff.reshape([-1,3])[ridxs_int[ridx]]
        barycentric.reshape([-1,3])[ridxs_out[ridx]] = barycentric.reshape([-1,3])[ridxs_int[ridx]]
        visibility.ravel()[ridxs_out[ridx]] = visibility.ravel()[ridxs_int[ridx]]
        xdiff.reshape([-1,3])[ridxs_int[ridx]] = dxlr.reshape([-1,3])[ridxs_int[ridx]]
        # visible.ravel()[ridxs_out[ridx]] = True

        tidx = tintGrad[:,0,0] > 0.0001
        ydiff.reshape([-1,3])[tidxs_out[tidx]] = ydiff.reshape([-1,3])[tidxs_int[tidx]]
        barycentric.reshape([-1,3])[tidxs_out[tidx]] = barycentric.reshape([-1,3])[tidxs_int[tidx]]
        visibility.ravel()[tidxs_out[tidx]] = visibility.ravel()[tidxs_int[tidx]]
        ydiff.reshape([-1,3])[tidxs_int[tidx]] = dybt.reshape([-1,3])[tidxs_int[tidx]]
        # visible.ravel()[tidxs_out[tidx]] = True

        bidx = bintGrad[:,0,0] < -0.0001
        ydiff.reshape([-1,3])[bidxs_out[bidx]] = ydiff.reshape([-1,3])[bidxs_int[bidx]]
        barycentric.reshape([-1,3])[bidxs_out[bidx]] = barycentric.reshape([-1,3])[bidxs_int[bidx]]
        visibility.ravel()[bidxs_out[bidx]] = visibility.ravel()[bidxs_int[bidx]]
        ydiff.reshape([-1,3])[bidxs_int[bidx]] = dytb.reshape([-1,3])[bidxs_int[bidx]]
        # visible.ravel()[bidxs_out[bidx]] = True

        # import ipdb
        # ipdb.set_trace()

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]

        n_channels = np.atleast_3d(observed).shape[2]
        shape = visibility.shape

        #2: Take the data and copy the corresponding dxs and dys to these new pixels.

        # Step 1: get the structure ready, ie the IS and the JS
        IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
        JS = col(f[visibility.ravel()[visible]].ravel())
        JS = np.hstack((JS*2, JS*2+1)).ravel()

        pxs = np.asarray(visible % shape[1], np.int32)
        pys = np.asarray(np.floor(np.floor(visible) / shape[1]), np.int32)

        if n_channels > 1:
            IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
            JS = np.concatenate([JS for i in range(n_channels)])

        datas = []

        # The data is weighted according to barycentric coordinates
        bc0 = col(barycentric[pys, pxs, 0])
        bc1 = col(barycentric[pys, pxs, 1])
        bc2 = col(barycentric[pys, pxs, 2])
        for k in range(n_channels):
            dxs = xdiff[pys, pxs, k]
            dys = ydiff[pys, pxs, k]
            if f.shape[1] == 3:
                datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1,col(dxs)*bc2,col(dys)*bc2)).ravel())
            else:
                datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1)).ravel())

        data = np.concatenate(datas)

        ij = np.vstack((IS.ravel(), JS.ravel()))

        result = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

        return result

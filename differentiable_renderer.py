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


class DifferentiableRenderer(Ch):
    terms = ['renderer', 'params_list']
    dterms = ['params']

    def compute_r(self):
        return self.renderer.r

    def compute_dr_wrt(self, wrt):
        for param in self.params_list:
            if wrt is param:
                return self.gradient_pred(wrt)
        else:
            return self.renderer.dr_wrt(wrt)


    def gradients(self):
        # self._call_on_changed()

        observed = self.renderer.r
        boundaryid_image = self.renderer.boundaryid_image
        barycentric = self.renderer.barycentric_image
        visibility = self.renderer.visibility_image
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        return self.dImage_wrt_2dVerts_bnd_gradient(observed, barycentric, observed.shape[0], observed.shape[1], boundaryid_image != 4294967295)

    def dImage_wrt_2dVerts_bnd_gradient(self, observed, barycentric, image_width, image_height, bnd_bool):

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
        dxrl = -np.hstack([np.diff(observed, n=1, axis=1), np.zeros([observed.shape[0],1,3])])

        dytb = np.vstack([np.zeros([1,observed.shape[1],3]), np.flipud(np.diff(np.flipud(observed), n=1, axis=0))])
        dxlr = np.hstack([np.zeros([observed.shape[0],1,3]),np.fliplr(np.diff(np.fliplr(observed), n=1, axis=1))])

        bary_sl = np.roll(barycentric , shift=-1, axis=1)
        bary_sr = np.roll(barycentric , shift=1, axis=1)
        bary_st = np.roll(barycentric , shift=-1, axis=0)
        bary_sb = np.roll(barycentric , shift=1, axis=0)

        return xdiff, ydiff, dybt, dxrl, dytb, dxlr, bary_sl, bary_sr, bary_st, bary_sb

    def gradient_pred(self, paramWrt):
        observed = self.renderer.r
        boundaryid_image = self.renderer.boundaryid_image
        barycentric = self.renderer.barycentric_image
        visibility = self.renderer.visibility_image
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        return self.dImage_wrt_2dVerts_predict(observed, paramWrt, visible, visibility, barycentric, observed.shape[0], observed.shape[1], self.renderer.v.shape[0], self.renderer.f, boundaryid_image != 4294967295)

    def boundary_neighborhood(self):
        boundary = self.renderer.boundaryid_image != 4294967295
        shape = boundary.shape

        notboundary = np.logical_not(boundary)
        horizontal = np.hstack((np.diff(boundary.astype(np.int8),axis=1), np.zeros((shape[0],1), dtype=np.int8)))
        # horizontal = np.hstack((np.diff(boundary.astype(np.int8),axis=1), np.zeros((shape[0],1), dtype=np.int8)))
        vertical = np.vstack((np.diff(boundary.astype(np.int8), axis=0), np.zeros((1,shape[1]), dtype=np.int8)))
        # vertical = np.vstack((np.diff(boundary.astype(np.int8), axis=0), np.zeros((1,shape[1]), dtype=np.int8)))

        pixr = (horizontal == 1)
        pixl = (horizontal == -1)
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

        camJac = self.renderer.camera.dr_wrt(paramWrt)

        xdiff, ydiff, dybt, dxrl, dytb, dxlr, bary_sl, bary_sr, bary_st, bary_sb = self.gradients()
        pixr, pixl, pixt, pixb, lidxs_out, ridxs_out, tidxs_out, bidxs_out, lidxs_int, ridxs_int, tidxs_int, bidxs_int = self.boundary_neighborhood()

        import ipdb
        ipdb.set_trace()

        #Where are triangles moving wrt to the image coordinates at the boundaries?
        lintGrad = camJac[f[visibility.ravel()[lidxs_int]]*2]
        rintGrad = camJac[f[visibility.ravel()[ridxs_int]]*2]
        tintGrad = camJac[f[visibility.ravel()[tidxs_int]]*2+1]
        bintGrad = camJac[f[visibility.ravel()[bidxs_int]]*2+1]

        # lintGrad[0] is a hack, should use barycentric at least.

        if lintGrad[0] < -0.001:
            xdiff.ravel()[lidxs_out] = xdiff[lidxs_int]
            barycentric.reshape([-1,3])[lidxs_out] = barycentric.reshape([-1,3])[lidxs_int]
            visibility.ravel()[lidxs_out] = visibility.ravel()[lidxs_int]
            xdiff.ravel()[lidxs_int] = dxrl[lidxs_int]
            visible.ravel()[lidxs_out] = True

        if rintGrad[0] > 0.001:
            xdiff.ravel()[ridxs_out] = xdiff[ridxs_int]
            barycentric.reshape([-1,3])[ridxs_out] = barycentric.reshape([-1,3])[ridxs_int]
            visibility.ravel()[ridxs_out] = visibility.ravel()[ridxs_int]
            xdiff.ravel()[ridxs_int] = dxlr[ridxs_int]
            visible.ravel()[ridxs_out] = True

        if tintGrad[0] > 0.001:
            ydiff.ravel()[tidxs_out] = ydiff[tidxs_int]
            barycentric.reshape([-1,3])[tidxs_out] = barycentric.reshape([-1,3])[tidxs_int]
            visibility.ravel()[tidxs_out] = visibility.ravel()[tidxs_int]
            ydiff.ravel()[tidxs_int] = dybt[tidxs_int]
            visible.ravel()[tidxs_out] = True

        if bintGrad[0] < -0.001:
            ydiff.ravel()[bidxs_out] = ydiff[bidxs_int]
            barycentric.reshape([-1,3])[bidxs_out] = barycentric.reshape([-1,3])[bidxs_int]
            visibility.ravel()[bidxs_out] = visibility.ravel()[bidxs_int]
            ydiff.ravel()[bidxs_int] = dytb[bidxs_int]
            visible.ravel()[bidxs_out] = True


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

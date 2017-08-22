# -*- coding: utf-8 -*-
"""
A parent and children classes for implementing matching
functionals for images. (Pseudo) distance, residual (scalar multiple in
gradient) and force (gradient of matching functional) are provided.

Author: Greg M. Fleishman
"""

import numpy as np
import scipy.ndimage.filters as ndif
import scipy.interpolate as interp
import vcalc


class matcher:
    """A template class for image matching functionals"""

    def __init__(self, tp, n):
        if tp == 'SSD' or tp == 'ssd':
            self._m = SSD()
        elif tp == 'GCC' or tp == 'gcc':
            self._m = GCC()
        elif tp == 'LCC' or tp == 'lcc':
            self._m = LCC(n)
        elif tp == 'MI' or tp == 'mi':
            self._m = MI(n)
        else:
            print 'matcher type must be SSD, GCC, LCC, or MI'

    def dist(self, ref, tmp):
        """compute distance between ref and tmp"""

        return self._m.dist(ref, tmp)

    def residual(self, ref, tmp):
        """compute gradient of distant w.r.t. template image"""

        return self._m.residual(ref, tmp)

    def force(self, ref, tmp, vox):
        """compute gradient of distance w.r.t. transformation"""

        r = self.residual(ref, tmp)[..., np.newaxis]
        gradTmp = vcalc.gradient(tmp, vox)
        return r * gradTmp


class SSD(matcher):
    """Sum of squared differences matcher"""

    def __init__(self):
        """Initialization of an SSD matcher requires no inputs"""

        return

    def dist(self, ref, tmp):
        """Compute the sum of squared differences between two images"""

        return 0.5*np.sum((ref - tmp)**2)

    def residual(self, ref, tmp):
        """Compute the gradient of SSD w.r.t. the template image"""

        return ref - tmp


class GCC(matcher):
    """Global Correlation Coefficient matcher"""

    def __init__(self):
        """Initialization of a GCC matcher requires no inputs"""

        return

    def dist(self, ref, tmp):
        """Compute the global correlation coefficient between two images"""

        # ignore background voxels
        mtmp = np.ma.masked_values(tmp, 0.0)
        mref = np.ma.masked_values(ref, 0.0)
        mtmp.mask *= mref.mask
        mref.mask = mtmp.mask
        # compute GCC
        u1 = np.mean(mtmp)
        v1 = np.var(mtmp)
        u2 = np.mean(mref)
        v2 = np.var(mref)
        v12 = np.mean((mtmp-u1)*(mref-u2))
        return v12**2/(v1*v2)

    def residual(self, ref, tmp):
        """Compute the gradient of GCC w.r.t. the template image"""

        # ignore background voxels
        mtmp = np.ma.masked_values(tmp, 0.0)
        mref = np.ma.masked_values(ref, 0.0)
        mtmp.mask *= mref.mask
        mref.mask = mtmp.mask
        # compute gradient
        u1 = np.mean(mtmp)
        v1 = np.var(mtmp)
        u2 = np.mean(mref)
        v2 = np.var(mref)
        v12 = np.mean((mtmp-u1)*(mref-u2))
        res = 2*v12*((mref-u2)*v1 - (mtmp-u1)*v12)/(v2*v1**2)
        return res.data


class LCC(matcher):
    """Local Correlation Coefficient matcher"""

    def __init__(self, n):
        """initialization of an LCC matcher requires no inputs"""

        self.n = n
        return

    def dist(self, ref, tmp):
        """compute the local correlation coefficient between two images"""

        u1, v1, u2, v2, v12 = self.get_moving_stats(ref, tmp)
        cc = v12**2/(v1*v2)
        # ignore background voxels and where CCL is undefined
        fill = np.isnan(cc) + np.isinf(cc) + ((tmp == 0) * (ref == 0))
        cc[fill] = 0
        return np.sum(cc)/np.sum(~fill)

    def residual(self, ref, tmp):
        """compute the gradient of LCC w.r.t. the template image"""

        u1, v1, u2, v2, v12 = self.get_moving_stats(ref, tmp)
        res = 2*v12*((ref-u2)*v1 - (tmp-u1)*v12)/(v2*v1**2)
        # background voxels don't need residual
        fill = np.isnan(res) + np.isinf(res) + ((tmp == 0) * (ref == 0))
        res[fill] = 0
        return res

    def get_moving_stats(self, ref, tmp):
        """compute local mean, var, and covar of reference and tmp images"""

        # Pad arrays with zeros
        d = len(tmp.shape)
        rad = (self.n - 1)/2
        tmpa = np.pad(tmp, [(rad+1, rad)]*d, mode='constant')
        refa = np.pad(ref, [(rad+1, rad)]*d, mode='constant')
        tmp2a = tmpa*tmpa
        ref2a = refa*refa
        reftmpa = refa*tmpa

        # get cumulative sums of arrays
        for i in range(d):
            tmpa = tmpa.cumsum(axis=i)
            refa = refa.cumsum(axis=i)
            tmp2a = tmp2a.cumsum(axis=i)
            ref2a = ref2a.cumsum(axis=i)
            reftmpa = reftmpa.cumsum(axis=i)

        # construct appropriate slice objects, compute partial sum arrays
        so1 = slice(None, -self.n, None)
        so2 = slice(self.n, None, None)
        so = [so2]*d
        u1 = np.copy(tmpa[so])
        u2 = np.copy(refa[so])
        v1 = np.copy(tmp2a[so])
        v2 = np.copy(ref2a[so])
        v12 = np.copy(reftmpa[so])
        bs = [0]*d
        for i in range(2**d - 1):

            # construct slice object corresponding to binary string
            p = 0
            for j in range(d):
                if bs[j] == 0:
                    so[j] = so1
                else:
                    so[j] = so2
                    p += 1

            s = (-1)**(d-p)
            u1 += s*tmpa[so]
            u2 += s*refa[so]
            v1 += s*tmp2a[so]
            v2 += s*ref2a[so]
            v12 += s*reftmpa[so]

            # increment binary string
            done = False
            b = d
            while not done and b > 0:
                if bs[b-1] == 0:
                    bs[b-1] = 1
                    done = True
                else:
                    bs[b-1] = 0
                b -= 1

        # normalize to window size to get average images
        w = 1.0/self.n**d
        u1 *= w
        u2 *= w
        v1 *= w
        v2 *= w
        v12 *= w

        # compute variances and covariance
        v1 -= u1*u1
        v2 -= u2*u2
        v12 -= u1*u2

        return u1, v1, u2, v2, v12


class MI(matcher):
    """Mutual Information matcher"""

    bins = 256          # User should have control over these
    sig = 2.5
    tmp_low_t = None
    tmp_high_t = None
    tmp_step = None
    e1 = None
    ref_low_t = None
    ref_high_t = None
    ref_step = None
    e2 = None

    def __init__(self):
        """initialization of an MI matcher requires no inputs"""

        return

    def dist(self, ref, tmp):
        """compute the mutual information between two images"""

        p1, p2, p12, p1p2, e1, e2 = self.getDistributions(ref, tmp)
        kld = p12 * np.log(p12/p1p2)
        kld[np.isnan(kld)] = 0
        kld[np.isinf(kld)] = 0
        return np.sum(kld)

    def residual(self, ref, tmp):
        """compute the gradient of MI w.r.t. the template image"""

        p1, p2, p12, p1p2, e1, e2 = self.getDistributions(ref, tmp)
        L = np.log(p12/p1p2)
        L[np.isnan(L)] = 0
        L[np.isinf(L)] = 0
        L = interp.RectBivariateSpline(e1[:-1], e2[:-1], L)
        res = L(tmp, ref, dx=1, grid=False)
        mask = ((tmp == 0) * (ref == 0))
        res[mask] = 0.0
        return res

    def getDistributions(self, ref, tmp):
        """compute the joint and marginal intensity histograms for images"""

        mask = ~((tmp == 0) * (ref == 0))
        if self.e1 is None or self.e2 is None:
            self.thresholdEdges(ref, tmp, mask)
        p12, e1, e2 = np.histogram2d(tmp[mask],
                                     ref[mask],
                                     bins=[self.e1, self.e2])
        p12 = ndif.gaussian_filter(p12, self.sig, mode='constant')
        p1 = np.sum(p12, axis=0)
        p2 = np.sum(p12, axis=1)
        p1mg, p2mg = np.meshgrid(p1, p2, indexing='ij')
        p1p2 = p1mg*p2mg
        p1 *= 1./np.sum(p1)
        p2 *= 1./np.sum(p2)
        p12 *= 1./np.sum(p12)
        p1p2 *= 1./np.sum(p1p2)

        return p1, p2, p12, p1p2, e1, e2

    def thresholdEdges(self, ref, tmp, mask):
        """determine upper and lower thresholds on intensity histograms

        TODO: this could possibly be solved by just storing the thresholds
        themselves and using parameters to the histogram functions"""

        p1, e1 = np.histogram(tmp[mask], bins=self.bins)
        p1 = p1.astype(np.float)
        p1 *= 1./np.sum(p1)
        p1cdf = np.cumsum(p1)
        low_t = e1[np.argmax(p1cdf > .0001)]
        high_t = e1[np.argmax(p1cdf > .9999)]
        new_e1 = np.empty(self.bins+1)
        step = (high_t - low_t)/(self.bins - 1)
        if low_t == 0:
            new_e1[0:-2] = np.arange(low_t, high_t-1e-6, step)
        else:
            new_e1[1:-1] = np.arange(low_t, high_t-1e-6, step)
        new_e1[0] = tmp.min()
        new_e1[-2] = high_t
        new_e1[-1] = tmp.max()
        self.tmp_low_t = low_t
        self.tmp_high_t = high_t
        self.tmp_step = step
        self.e1 = new_e1

        p2, e2 = np.histogram(ref[mask], bins=self.bins)
        p2 = p2.astype(np.float)
        p2 *= 1./np.sum(p2)
        p2cdf = np.cumsum(p2)
        low_t = e2[np.argmax(p2cdf > .0001)]
        high_t = e2[np.argmax(p2cdf > .9999)]
        new_e2 = np.empty(self.bins+1)
        step = (high_t - low_t)/(self.bins - 1)
        if low_t == 0:
            new_e2[0:-2] = np.arange(low_t, high_t-1e-6, step)
        else:
            new_e2[1:-1] = np.arange(low_t, high_t-1e-6, step)
        new_e2[0] = ref.min()
        new_e2[-2] = high_t
        new_e2[-1] = ref.max()
        self.ref_low_t = low_t
        self.ref_high_t = high_t
        self.ref_step = step
        self.e2 = new_e2

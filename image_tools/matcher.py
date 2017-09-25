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

    bounds = None
    slope = None
    intercept = None

    def __init__(self, tp, n):
        if tp == 'SSD' or tp == 'ssd':
            self._m = SSD()
        elif tp == 'GCC' or tp == 'gcc':
            self._m = GCC()
        elif tp == 'LCC' or tp == 'lcc':
            self._m = LCC(n)
        elif tp == 'MI' or tp == 'mi':
            self._m = MI(n)

    def dist(self, ref, tmp, normalized=True):
        """compute distance between ref and tmp"""

        return self._m.dist(ref, tmp, normalized)

    def residual(self, ref, tmp):
        """compute gradient of distant w.r.t. template image
        normalizes range to be between +/- 1
        range is [.5, 99.5] percentiles, for robustness to outliers
        TODO: make this residual damping param available to user"""

        res = self._m.residual(ref, tmp)
        if self.slope is None or self.intercept is None:
            bounds = np.percentile(res[res != 0], [.5, 99.5])
            self.slope = 2./(bounds[1] - bounds[0])
            self.intercept = 1. - self.slope * bounds[1]
            self.bounds = bounds
        res[res < self.bounds[0]] = self.bounds[0]
        res[res > self.bounds[1]] = self.bounds[1]
        res = self.slope * res + self.intercept
        return res

    def force(self, ref, tmp, vox):
        """compute gradient of distance w.r.t. transformation"""

        r = self.residual(ref, tmp)[..., np.newaxis]
        gradTmp = vcalc.gradient(tmp, vox)
        return r * gradTmp


class SSD(matcher):
    """Sum of squared differences matcher"""
    
    initial_distance = None

    def __init__(self):
        """Initialization of an SSD matcher requires no inputs"""

        return

    def dist(self, ref, tmp, normalized):
        """Compute the sum of squared differences between two images"""

        distance = 0.5*np.sum((ref - tmp)**2)
        if self.initial_distance is None and distance != 0:
            self.initial_distance = distance
        if normalized and distance != 0:
            distance *= 1./self.initial_distance
        return distance

    def residual(self, ref, tmp):
        """Compute the gradient of SSD w.r.t. the template image"""

        r = ref - tmp
        r[tmp == 0] = 0
        return r


# TODO: rewrite using input mask instead of masked arrays
class GCC(matcher):
    """Global Correlation Coefficient matcher"""
    
    initial_distance = None

    def __init__(self):
        """Initialization of a GCC matcher requires no inputs"""

        return

    def dist(self, ref, tmp, normalized):
        """Compute the global correlation coefficient between two images"""

        # ignore background voxels
        mtmp = np.ma.masked_values(tmp, 0.0)
        mref = np.ma.masked_values(ref, 0.0)
        # compute GCC
        u1 = np.mean(mtmp)
        v1 = np.var(mtmp)
        u2 = np.mean(mref)
        v2 = np.var(mref)
        v12 = np.mean((mtmp-u1)*(mref-u2))
        distance = 1. - v12**2/(v1*v2)
        if self.initial_distance is None and distance != 0:
            self.initial_distance = distance
        if normalized and distance != 0:
            distance *= 1./self.initial_distance
        return distance

    def residual(self, ref, tmp):
        """Compute the gradient of GCC w.r.t. the template image"""

        # ignore background voxels
        mtmp = np.ma.masked_values(tmp, 0.0)
        mref = np.ma.masked_values(ref, 0.0)
        # compute gradient
        u1 = np.mean(mtmp)
        v1 = np.var(mtmp)
        u2 = np.mean(mref)
        v2 = np.var(mref)
        v12 = np.mean((mtmp-u1)*(mref-u2))
        res = 2*v12*((mref-u2)*v1 - (mtmp-u1)*v12)/(v2*v1**2)
        res[tmp == 0] = 0
        return res.data


class LCC(matcher):
    """Local Correlation Coefficient matcher"""
    
    initial_distance = None

    def __init__(self, n):
        """initialization of an LCC matcher requires no inputs"""

        self.n = n
        return

    def dist(self, ref, tmp, normalized):
        """compute the local correlation coefficient between two images"""

        u1, v1, u2, v2, v12 = self._get_moving_stats(ref, tmp)
        cc = v12**2/(v1*v2)
        # ignore background voxels and where CCL is undefined
        fill = np.isnan(cc) + np.isinf(cc) + (tmp == 0)
        cc[fill] = 0
        distance = 1. - np.sum(cc)/np.sum(~fill)
        if self.initial_distance is None and distance != 0:
            self.initial_distance = distance
        if normalized and distance != 0:
            distance *= 1./self.initial_distance
        return distance

    def residual(self, ref, tmp):
        """compute the gradient of LCC w.r.t. the template image"""

        u1, v1, u2, v2, v12 = self._get_moving_stats(ref, tmp)
        res = 2*v12*((ref-u2)*v1 - (tmp-u1)*v12)/(v2*v1**2)
        # ignore background voxels and where CCL is undefined
        fill = np.isnan(res) + np.isinf(res) + (tmp == 0)
        res[fill] = 0
        return res

    def _get_moving_stats(self, ref, tmp):
        """compute local mean, var, and covar of reference and tmp images"""

        # Pad arrays, compute summed area tables
        d = len(tmp.shape)
        rad = (self.n - 1)/2
        tmpa = np.pad(tmp, [(rad, rad)]*d, mode='constant')
        refa = np.pad(ref, [(rad, rad)]*d, mode='constant')
        sats = [tmpa, tmpa*tmpa, refa, refa*refa, refa*tmpa]
        for i in range(d):
            sats = [sat.cumsum(axis=i, out=sat) for sat in sats]

        # construct appropriate slice objects, compute partial sum arrays
        so = self._get_slice_object([1]*d)
        stats = [np.copy(sat[so]) for sat in sats]
        bs = [0]*d
        for i in range(2**d - 1):
            # do appropriate arithmetic
            so = self._get_slice_object(bs)
            s = (-1)**(d-np.sum(bs))
            for i in range(5):
                stats[i] += s*sats[i][so]
            # increment binary string
            bs = self._increment_binary_string(bs)

        # normalize to window size to get average images
        stats = [stat * (1./self.n**d) for stat in stats]
        # compute variances and covariance
        stats[1] -= stats[0]*stats[0]
        stats[3] -= stats[2]*stats[2]
        stats[4] -= stats[0]*stats[2]

        return stats[0], stats[1], stats[2], stats[3], stats[4]

    def _get_slice_object(self, bs):
        # construct slice object corresponding to binary string
        so1 = slice(None, -self.n+1, None)
        so2 = slice(self.n-1, None, None)
        so = []
        for bit in bs:
            if bit == 0:
                so.append(so1)
            else:
                so.append(so2)
        return so

    def _increment_binary_string(self, bs):
        # increment binary string
        done = False
        b = len(bs)
        while not done and b > 0:
            if bs[b-1] == 0:
                bs[b-1] = 1
                done = True
            else:
                bs[b-1] = 0
            b -= 1
        return bs


# TODO: something bizarre is still happening with this functional
# TODO: consider some kind of local MI (histograms computed globally)
# but residual at a voxel is informed by patch, not just voxel
class MI(matcher):
    """Mutual Information matcher"""

    tmp_entropy = None
    ref_entropy = None
    initial_distance = None
    bins = 256  # TODO: user could have control over this
    sig = 7.5  # TODO: user could have control over this
    e1 = None
    e2 = None

    def __init__(self, n):
        """initialization of an MI matcher requires no inputs"""

        return

    def dist(self, ref, tmp, normalized):
        """compute the mutual information between two images"""

        p1, p2, p12, p1p2 = self._get_distributions(ref, tmp)
        if self.tmp_entropy is None or self.ref_entropy:
            self.tmp_entropy = - np.sum(p1 * np.log(p1))
            self.ref_entropy = - np.sum(p2 * np.log(p2))
        kld = p12 * np.log(p12/p1p2)
        kld[np.isnan(kld)] = 0
        kld[np.isinf(kld)] = 0
        distance = self.ref_entropy - np.sum(kld)
        if np.allclose(ref, tmp):
            distance = 0
        else:
            if self.initial_distance is None:
                self.initial_distance = distance
            if normalized:
                distance *= 1./self.initial_distance
        return distance

    def residual(self, ref, tmp):
        """compute the gradient of MI w.r.t. the template image"""

        p1, p2, p12, p1p2 = self._get_distributions(ref, tmp)
        L = np.log(p12/p1p2)
        L = interp.RectBivariateSpline(self.e1[:-1], self.e2[:-1], L, s=256**2)
        res = L(tmp, ref, dx=1, grid=False)
        res[tmp == 0] = 0.0
        return res

    def _get_distributions(self, ref, tmp):
        """compute the joint and marginal intensity histograms for images"""

        tmpmask = (tmp != 0)
        refmask = (ref != 0)
        joint_mask = tmpmask * refmask
        p12, self.e1, self.e2 = np.histogram2d(tmp[joint_mask],
                                     ref[joint_mask],
                                     bins=self.bins)
        p12 = ndif.gaussian_filter(p12, self.sig, mode='constant')
        p1 = np.sum(p12, axis=1)  # sum over intensities in ref to marginalize
        p2 = np.sum(p12, axis=0)  # sum over intensities in tmp to marginzlize
        p1mg, p2mg = np.meshgrid(p1, p2, indexing='ij')
        p1p2 = p1mg*p2mg
        p1 *= 1./np.sum(p1)
        p2 *= 1./np.sum(p2)
        p12 *= 1./np.sum(p12)
        p1p2 *= 1./np.sum(p1p2)

        return p1, p2, p12, p1p2

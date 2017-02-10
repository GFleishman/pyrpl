# -*- coding: utf-8 -*-
"""
A short class for resampling and applying spatial transformations to images
and vector fields.

Author: Greg M. Fleishman
"""

import numpy as np
import scipy.ndimage.interpolation as ndii


class transformer:
    """A class to resample and apply transformations to images

    An instance of transformer requires no initialization. It stores a
    position array at the resolution and voxel size of the most recent
    image/vector field it resampled or warped."""

    def __init__(self):
        """Initialize an empty position array"""

        self.X = np.array([])

    def new_vox_size(self, sh1, sh2, vox):
        """Compute voxel size when grid goes from resolution sh1 to sh2"""

        sh1 = np.array(sh1).astype(np.float)
        sh2 = np.array(sh2).astype(np.float)
        return vox*sh1/sh2

    def position_array(self, sh, vox):
        """Return a position array in physical coordinates with shape sh"""

        if sh == self.X.shape[:-1]:
            pass
        else:
            d = len(sh)
            X = np.empty(sh + (d,))
            sha = np.diag(sh) - np.identity(d) + 1
            oa = np.ones(sh)
            for i in range(d):
                X[..., i] = np.reshape(np.arange(sh[i]), sha[i])*oa*vox[i]
            self.X = X
        return np.copy(self.X)

    def resample(self, img, vox, res, vec=False):
        """Resample img to resolution res, compute new voxel size"""

        if not vec:
            img = img[..., np.newaxis]

        if img.shape[:-1] == res:
            return img.squeeze()
        else:
            nvox = self.new_vox_size(img.shape[:-1], res, vox)
            X = self.position_array(res, nvox)/vox

            ret = np.empty(res + (img.shape[-1],))
            for i in range(img.shape[-1]):
                ret[..., i] = self.interpolate(img[..., i], X)
            return ret.squeeze()

    def apply_transform(self, img, vox, u, vec=False):
        """Return img warped by transformation u

        if vec is True, img is a vector field"""

        if not vec:
            img = img[..., np.newaxis]
        X = u * 1./vox
        ret = np.empty(u.shape[:-1] + (img.shape[-1],))
        for i in range(img.shape[-1]):
            ret[..., i] = self.interpolate(img[..., i], X)
        return ret.squeeze()

    def interpolate(self, img, X, o=1):
        """Return img linearly interpolated at positions X"""

        return ndii.map_coordinates(img, np.rollaxis(X, -1), order=o,
                                    mode='wrap', prefilter=False)

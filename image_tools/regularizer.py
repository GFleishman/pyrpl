# -*- coding: utf-8 -*-
"""
Regularization kernel classes including a template class

Author: Greg M. Fleishman
"""

import numpy as np
import scipy.ndimage.filters as ndif


class regularizer:
    """A template class for regularizers"""

    def __init__(self, tp, a, b, c, d, vox, sh=None):
        """Initialize the correct type of regularizer"""

        if tp == 'gaussian':
            self._r = gaussian(a, vox)
        elif tp == 'differential':
            self._r = differential(a, b, c, d, vox, sh)
        else:
            print "regularizer type must be gaussian or differential"

    def regularize(self, f):
        return self._r.regularize(f)

    def convolve(self, f):
        return self._r.convolve(f)


class gaussian(regularizer):
    """gaussian regularizer"""

    def __init__(self, a, vox):
        # adapt a from physical to voxel units
        self.sig = a * (1.0/vox)

    def regularize(self, f):
        F = np.empty_like(f)
        for i in range(f.shape[-1]):
            F[..., i] = ndif.gaussian_filter(f[..., i], self.sig)
        return F


class differential(regularizer):
    """differential operator regularizer"""

    def __init__(self, a, b, c, d, vox, sh):
        """Explicit construction of the differential operator's DFT"""

        # cast and store parameters
        self.a = np.float(a)
        self.b = np.float(b)
        self.c = np.float(c)
        self.d = np.float(d)

        # define some useful ingredients for later
        dim = len(sh)
        sha = np.diag(sh) + np.ones((dim, dim)) - np.identity(dim)
        oa = np.ones(sh)

        # if grad of div term is 0, kernel is a scalar, else a Lin Txm
        if b == 0.0:
            self.L = oa*self.c
        else:
            self.L = np.zeros(sh + (dim, dim)) + np.identity(dim)*self.c

        # compute the scalar (or diagonal) term(s) of kernel
        for i in range(dim):
            q = np.fft.fftfreq(sh[i], d=vox[i])
            X = 1 - np.cos(q*2.0*np.pi*vox[i])
            X *= 2*self.a/vox[i]**2
            X = np.reshape(X, sha[i])*oa
            if b == 0.0:
                self.L += X
            else:
                for j in range(dim):
                    self.L[..., j, j] += X
                self.L[..., i, i] += self.b*X/self.a

        # compute off diagonal terms of kernel
        if b != 0.0:
            for i in range(dim):
                for j in range(i+1, dim):
                    q = np.fft.fftfreq(sh[i], d=vox[i])
                    X = np.sin(q*2.0*np.pi*vox[i])
                    X1 = np.reshape(X, sha[i])*oa
                    q = np.fft.fftfreq(sh[j], d=vox[j])
                    X = np.sin(q*2.0*np.pi*vox[j])
                    X2 = np.reshape(X, sha[j])*oa
                    X = X1*X2*self.b/(vox[i]*vox[j])
                    self.L[..., i, j] = X
                    self.L[..., j, i] = X

        # I only need half the coefficients (because we're using rfft)
        # compute and store the inverse kernel for regularization
        if b == 0.0:
            self.L = self.L[..., :sh[-1]/2+1]**self.d
            self.K = self.L**-1.0
            self.L = self.L[..., np.newaxis]
            self.K = self.K[..., np.newaxis]
        else:
            self.L = self.L[..., :sh[-1]/2+1, :, :]
            cp = np.copy(self.L)
            for i in range(int(self.d-1)):
                self.L = np.einsum('...ij,...jk->...ik', self.L, cp)
            self.K = _gu_pinv(self.L)

    def regularize(self, f):
        """Apply inverse kernel to vector field to regularize"""

        F = _fft(f)
        if self.b == 0:
            U = self.K*F
        else:
            F = np.reshape(F, F.shape + (1,))
            U = np.einsum('...ij,...jk->...ik', self.K, F)
            U = np.reshape(U, U.shape[:-1])
        return _ifft(U, f.shape)

    def convolve(self, f):
        """Apply the kernel to vector field"""

        F = _fft(f)
        if self.b == 0:
            U = self.L*F
        else:
            F = np.reshape(F, F.shape + (1,))
            U = np.einsum('...ij,...jk->...ik', self.L, F)
            U = np.reshape(U, U.shape[:-1])
        return _ifft(U, f.shape)


def _gu_pinv(a, rcond=1e-15):
    """Return the pseudo-inverse of matrices at every voxel"""

    a = np.asarray(a)
    swap = np.arange(a.ndim)
    swap[[-2, -1]] = swap[[-1, -2]]
    u, s, v = np.linalg.svd(a)
    cutoff = np.maximum.reduce(s, axis=-1, keepdims=True) * rcond
    mask = s > cutoff
    s[mask] = 1. / s[mask]
    s[~mask] = 0

    return np.einsum('...uv,...vw->...uw',
                     np.transpose(v, swap) * s[..., None, :],
                     np.transpose(u, swap))


def _fft(f):
    """Return the DFT of the real valued vector field f"""

    sh = f.shape[:-1]
    d = f.shape[-1]
    F = np.empty(sh[:-1] + (sh[-1]/2+1, d), dtype='cfloat')
    for i in range(0, d):
        F[..., i] = np.fft.rfftn(f[..., i])
    return F


def _ifft(F, sh):
    """Return the iDFT of the vector field F"""

    f = np.empty(sh, dtype='float64')
    for i in range(0, sh[-1]):
        f[..., i] = np.fft.irfftn(F[..., i], s=sh[:-1])
    return f

# -*- coding: utf-8 -*-
"""
A small module for computing vector calculus derivatives on regular
grids using central finite differences.

Author: Greg M. Fleishman
"""

import numpy as np
import scipy.ndimage.filters as ndif


def partial(img, vox, axis, mode='wrap'):
    """Compute partial derivative of img w.r.t. axis direction"""
    d = len(img.shape)
    shm = np.identity(d)*2 + 1
    shm = shm.astype(np.int)
    mask = np.array([1.0, 0.0, -1.0])
    m = np.reshape(mask, shm[axis])/(2.0*vox[axis])
    return ndif.convolve(img, m, mode=mode)


def gradient(img, vox):
    """Compute the gradient of img"""

    sh = img.shape
    d = len(sh)
    gra = np.empty(sh + (d,))

    for i in range(d):
        gra[..., i] = partial(img, vox, axis=i)
    return gra


def jacobian(img, vox, txm=True):
    """Compute the Jacobian of the vector field img

    For transformations, this assumes the Jacobian on the boundary is
    identity"""

    sh = img.shape
    d = len(sh[:-1])
    m = sh[-1]
    jac = np.empty(sh[:-1] + (m, d))

    for i in range(m):
        jac[..., i, :] = gradient(img[..., i], vox)
    if txm:
        beg = slice(0, 1, None)
        end = slice(-1, None, None)
        for i in range(d):
            so = [slice(None, None, None)]*d
            so[i] = beg
            jac[so] = np.identity(m)
            so[i] = end
            jac[so] = np.identity(m)
    return jac


def divergence(img, vox):
    """Return the divergence of the vector field img"""

    sh = img.shape[:-1]
    d = len(sh)
    div = np.zeros(sh)

    for i in range(d):
        div += partial(img[..., i], vox, axis=i)
    return div

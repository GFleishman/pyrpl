# -*- coding: utf-8 -*-
"""
Author: Greg M. Fleishman

Description: Container classes used by geodesic shooting methods

Dependencies: NumPy
"""

import numpy as np


class LddmmRegistrationDataContainer:
    """A container for the resampled image time series and
    geodesic parameters"""

    # Full resolution variables
    J0 = None
    J1 = None

    # Lower resolution variables
    v = None
    uf = None
    ub = None
    If = None
    Ib = None

    # Other variables
    T = None
    t = None
    d = None
    full_res = None
    curr_res = None
    full_vox = None
    curr_vox = None

    # Parameter dictionary and optimization score
    params = None
    dist = None

    def __init__(self, J0, J1, T, params):
        self.J0 = J0
        self.J1 = J1
        self.T = T
        self.params = params

        self.full_res = J0.shape
        self.curr_res = J0.shape
        self.full_vox = params['vox']
        self.curr_vox = params['vox']
        self.d = len(self.full_res)

        self.t = self.compute_t(T, params['h'])
        self.dist = np.empty((params['h'], np.sum(params['its'])))

        self.v = np.zeros((params['h'],) + self.curr_res + (3,))
        self.uf = np.zeros(self.curr_res + (3,))
        self.ub = np.zeros(self.curr_res + (3,))

        self.If = np.zeros((params['h'],) + self.curr_res)
        self.Ib = np.zeros((params['h'],) + self.curr_res)

    def resample(self, res, _t):
        sh = self.v.shape
        v = np.copy(self.v)
        self.v = np.empty((sh[0],) + res + (3,))
        for i in range(sh[0]):
            for j in range(3):
                self.v[i, ..., j] = _t.resample(v[i, ..., j],
                                                self.curr_vox,
                                                res)

        self.uf = np.zeros(res + (3,))
        self.ub = np.zeros(res + (3,))
        self.If = np.zeroes((sh[0],) + res)
        self.Ib = np.zeroes((sh[0],) + res)

        self.curr_res = res
        self.curr_vox = _t.getNewVoxSize(self.full_res,
                                         res,
                                         self.full_vox)

    def compute_t(self, T, h):
        """Compute time points along discrete sampling of geodesic"""
        t = ((T[-1] - T[0])/(h - 1))*np.arange(h) + T[0]

        for time in T:
            idx = min(range(h), key=lambda i: abs(t[i] - time))
            t[idx] = time
        return t
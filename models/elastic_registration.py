# -*- coding: utf-8 -*-

import numpy as np
import PyRPL.image_tools.transformer as transformer
import PyRPL.image_tools.matcher as matcher
import PyRPL.image_tools.regularizer_fftw as regularizer


class elastic_registration:

    def __init__(self, J, T, params):
        """Initialize image level tools"""

        self.dc = data_container(J, T, params)
        self._t = transformer.transformer()
        self._m = matcher.matcher(params['mType'])
        self._r = regularizer.regularizer(params['rType'],
                                          params['a'],
                                          params['b'],
                                          params['c'],
                                          params['d'],
                                          self.dc.curr_vox,
                                          self.dc.curr_res)

    def evaluate(self):
        """Evaluate the objective functional"""

        dc = self.dc
        if dc.curr_res != dc.full_res:
            uhr = self._t.resample(dc.u, dc.curr_vox, dc.full_res, vec=True)
            dc.I0 = self._t.apply_transform(dc.J1, dc.full_vox, uhr)
        else:
            dc.I0 = self._t.apply_transform(dc.J1, dc.full_vox, dc.u)

        # compute regularizer term
        u_energy = 10  # temporary, should be elastic potential energy of u
        # compute matching functional
        data_match = self._m.dist(dc.J0, dc.I0)
        return [u_energy, data_match]

    def get_gradient(self):
        """Obtain the gradient w.r.t. the transformation parameters"""

        dc = self.dc
        f = self._m.force(dc.I0, dc.J0, dc.full_vox)
        if dc.curr_res != dc.full_res:
            f = self._t.resample(f, dc.full_vox, dc.curr_res, vec=True)
        f = self._r.regularize(f)
        grad_mag = np.prod(dc.curr_res) * np.sum(f * f)
        return [f, grad_mag]

    def take_step(self, update):
        """Take an optimization step"""

        self.dc.u += update

    def resample(self, res):
        """Resample all objects for multi-resolution schemes"""

        self.dc.resample(res, self._t)
        self._r = regularizer.regularizer(self.dc.params['rType'],
                                          self.dc.params['a'],
                                          self.dc.params['b'],
                                          self.dc.params['c'],
                                          self.dc.params['d'],
                                          self.dc.curr_vox, res)


class data_container:
    """A container for elastic registration data"""

    def __init__(self, J, T, params):
        self.J0 = J[0]
        self.J1 = J[1]
        self.params = params

        self.I0 = np.copy(J[0])
        self.d = len(J[0].shape)
        self.full_res = J[0].shape
        self.curr_res = J[0].shape
        self.full_vox = params['vox']
        self.curr_vox = params['vox']

        self.u = np.empty(self.full_res + (self.d,))
        sha = np.diag(self.full_res) - np.identity(self.d) + 1
        oa = np.ones(self.full_res)
        for i in range(self.d):
            self.u[..., i] = np.reshape(np.arange(self.full_res[i]), sha[i])
            self.u[..., i] *= oa * self.curr_vox[i]

    def resample(self, res, _t):
        self.u = _t.resample(self.u, self.curr_vox, res, vec=True)
        self.curr_res = res
        self.curr_vox = _t.new_vox_size(self.full_res, res, self.full_vox)

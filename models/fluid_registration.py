# -*- coding: utf-8 -*-

import numpy as np
import PyRPL.image_tools.transformer as transformer
import PyRPL.image_tools.matcher as matcher
import PyRPL.image_tools.regularizer_fftw as regularizer
import PyRPL.image_tools.vcalc as vcalc


class fluid_registration:

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
            dc.I1 = self._t.apply_transform(dc.J0, dc.full_vox, uhr)
        else:
            dc.I1 = self._t.apply_transform(dc.J0, dc.full_vox, dc.u)

        # compute regularizer term
        u_energy = 10  # temporary, should be elastic potential energy of u
        # compute matching functional
        data_match = self._m.dist(dc.J1, dc.I1)
        return [u_energy, data_match]

    def get_gradient(self):
        """Obtain the gradient w.r.t. the transformation parameters"""

        dc = self.dc
        f = self._m.force(dc.J1, dc.I1, dc.full_vox)
        if dc.curr_res != dc.full_res:
            f = self._t.resample(f, dc.full_vox, dc.curr_res, vec=True)
        fr = self._r.regularize(f)
        grad_mag = np.prod(dc.curr_res) * np.sum(fr * fr)
        return [fr, grad_mag]

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

        self.I1 = np.copy(J[0])
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

        
class FluidRegistrationDataContainer:
    """A container for elastic registration data"""

    def __init__(self, J0, J1, params):
        self.J0 = J0
        self.J1 = J1
        self.params = params

        self.I1 = np.copy(J0)
        self.d = len(J0.shape)
        self.full_res = J0.shape
        self.curr_res = J0.shape
        self.full_vox = params['vox']
        self.curr_vox = params['vox']
        self.txm = np.empty(J0.shape + (self.d,))
        self.dist = np.empty(np.sum(params['its']))

        self.u = np.zeros(J0.shape + (self.d,))

    def resample(self, res, _t):
        u = np.copy(self.u)
        self.u = np.empty(res + (self.d,))
        for i in range(self.d):
            self.u[..., i] = _t.resample(u[..., i], self.curr_vox, res)
        self.flr = np.empty_like(self.u)

        self.curr_res = res
        self.curr_vox = _t.getNewVoxSize(self.full_res, res, self.full_vox)


def find_transformation(frdc):

    k = 0   # count the total iterations
    j = 0   # count the resamples
    while k < np.sum(frdc.params['its']):

        # resample when necessary
        if k == np.sum(frdc.params['its'][0:j]):

            frdc.resample(frdc.params['res'][j], _t)
            _r._initialize(frdc.params['a'],
                           frdc.params['b'],
                           frdc.params['c'],
                           frdc.params['d'],
                           frdc.curr_vox,
                           frdc.params['res'][j])

            # experimental
            frdc.params['uStep'] = frdc.params['uStep']/2.0

            j += 1

        # compute matching functional
        frdc.dist[k] = _m.dist(frdc.J1, frdc.I1)

        # optimize
        if k < np.sum(frdc.params['its'][:-1]):
            f = _m.force(frdc.J1, frdc.I1, frdc.full_vox)
            for i in range(frdc.d):
                frdc.flr[..., i] = _t.resample(f[..., i],
                                               frdc.full_vox,
                                               frdc.curr_res)

            frdc.v = _r.regularize(frdc.flr)
            J = vcalc.jacobian(frdc.u, frdc.curr_vox, txm=False)
            v = np.reshape(frdc.v, frdc.v.shape + (1,))
            J = frdc.v - np.einsum('...ij,...jk->...ik', J, v).squeeze()
            frdc.u = frdc.u + frdc.params['uStep']*J

            for i in range(frdc.d):
                frdc.txm[..., i] = _t.resample(frdc.u[..., i],
                                               frdc.curr_vox,
                                               frdc.full_res)

            frdc.I1 = _t.applyTransform(frdc.J0, frdc.full_vox, frdc.txm)
        else:
            f = _m.force(frdc.J1, frdc.I1, frdc.full_vox)
            frdc.v = _r.regularize(f)
            J = vcalc.jacobian(frdc.u, frdc.curr_vox, txm=False)
            v = np.reshape(frdc.v, frdc.v.shape + (1,))
            J = frdc.v - np.einsum('...ij,...jk->...ik', J, v).squeeze()
            frdc.u = frdc.u + frdc.params['uStep']*J
            frdc.I1 = _t.applyTransform(frdc.J0, frdc.full_vox, frdc.u)

        k += 1

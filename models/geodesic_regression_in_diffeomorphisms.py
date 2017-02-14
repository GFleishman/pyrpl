# -*- coding: utf-8 -*-
"""
The geodesic regression in diffeomorphisms model. Symmetric interleaved
geodesic shooting is available.

Author: Greg M. Fleishman
"""

# fundamental numerical imports
import numpy as np
import numpy.linalg as la

# fundamental image level imports
import pyrpl.image_tools.vcalc as vcalc
import pyrpl.image_tools.fvm as fvm
import pyrpl.image_tools.transformer as transformer
import pyrpl.image_tools.matcher as matcher
import pyrpl.image_tools.regularizer_fftw as regularizer


class geodesic_regression_in_diffeomorphisms:
    """Geodesic regression in diffeomorphisms nonlinear registration model

    This model fits a geodesic of diffeomorphisms, parameterized by a scalar
    initial momentum field, through the given time series of images."""

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

        return self.solveForward()

    def get_gradient(self):
        """Obtain the gradient w.r.t. the transformation parameters"""

        return self.solveBackward()

    def take_step(self, update):
        """Take an optimization step"""

        self.dc.P[0] += update

    def symmetrize(self, dc):
        """Implement symmetrization formulas

        These formulas are based on my IPMI 2017 paper:
        Symmetric Interleaved Geodesic Shooting in Diffeomorphisms"""

        # compute symmetric transforms
        uf_sym = np.empty_like(dc.uf)
        ub_sym = np.empty_like(dc.ub)
        uf_sym[0] = self._t.position_array(dc.curr_res, dc.curr_vox)
        ub_sym[0] = self._t.position_array(dc.curr_res, dc.curr_vox)
        uf_sym[-1] = dc.ub[-1]
        ub_sym[-1] = dc.uf[-1]
        for i in range(1, dc.params['h'] - 1):
            uf_sym[i] = self._t.apply_transform(dc.uf[-i-1], dc.curr_vox,
                                                dc.ub[-1], vec=True)
            ub_sym[i] = self._t.apply_transform(dc.uf[-1], dc.curr_vox,
                                                dc.ub[-i-1], vec=True)
        dc.uf = uf_sym
        dc.ub = ub_sym

        # swap image labels
        dc.J = dc.J[::-1]
        dc.Ifr[0] = dc.J[0]
        dc.txm = self._t.resample(dc.ub[-1],
                                  dc.curr_vox,
                                  dc.full_res,
                                  vec=True)
        dc.Ifr[-1] = self._t.apply_transform(dc.Ifr[0],
                                             dc.full_vox,
                                             dc.txm)
        # compute symmetric image path
        for i in range(dc.params['h']):
            dc.I[i] = self._t.apply_transform(dc.Ifr[0],
                                              dc.full_vox,
                                              dc.ub[i])
        # invert momentum
        dc.P = -dc.P[::-1]

    def solveForward(self):
        """Shoot the geodesic forward given initial conditions"""

        # cut down on all the self calls
        dc = self.dc
        # check cfl condition
        dc.satisfy_cfl()

        # compute initial velocity from initial momentum
        dI = vcalc.gradient(dc.I[0], dc.curr_vox)
        m = dI * dc.P[0][..., np.newaxis]
        dc.v = - self._r.regularize(m)

        # compute magnitude of initial momentum field
        P0_mag = - np.prod(dc.curr_vox) * np.sum(m * dc.v)
        P0_mag *= dc.params['rat']

        # Initial min number of time steps due to CFL
        dc.cfl_nums[0] = abs(dc.v * dc.T[-1]/dc.curr_vox).max()

        i = 1
        idx = 1
        while i < dc.params['h']:
            # Forward Euler
            dc.uf[i] = (dc.uf[i-1] + (dc.t[i] - dc.t[i-1]) *
                        self._t.apply_transform(
                        dc.v, dc.curr_vox, dc.uf[i-1], vec=True))

            # Advance backward transformation, ctu
            dc.ub[i] = fvm.solve_advection_ctu(dc.ub[i-1],
                                               dc.v,
                                               dc.curr_vox,
                                               dc.t[i] - dc.t[i-1],
                                               self._t)

            # Advance the image with group action
            dc.I[i] = self._t.apply_transform(dc.Ifr[0],
                                              dc.full_vox,
                                              dc.ub[i])

            # Advance the momentum with coadjoint transport
            jdet = la.det(vcalc.jacobian(dc.ub[i], dc.curr_vox))
            dc.P[i] = jdet * self._t.apply_transform(
                                   dc.P[0], dc.curr_vox, dc.ub[i])

            # Compute velocity from momentum
            dI = vcalc.gradient(dc.I[i], dc.curr_vox)
            dc.v = - self._r.regularize(dI * dc.P[i][..., np.newaxis])

            # Store new CFL min number of time steps
            dc.cfl_nums[i] = abs(dc.v * dc.T[-1]/dc.curr_vox).max()

            # Check if time point lines up with an image in the time series
            # if so, compute full resolution version
            if dc.t[i] in dc.T:
                dc.txm = self._t.resample(dc.ub[i],
                                          dc.curr_vox,
                                          dc.full_res,
                                          vec=True)
                dc.Ifr[idx] = self._t.apply_transform(dc.Ifr[0],
                                                      dc.full_vox,
                                                      dc.txm)
                idx += 1

            i += 1

        # compute image matching functionals
        obj_func = []
        for i in range(dc.N):
            obj_func.append(self._m.dist(dc.Ifr[i], dc.J[i]))

        # symmetrize the problem
#        self.symmetrize(dc)

        # return complete evaluation of objective function
        return [P0_mag] + obj_func

    def solveBackward(self):
        """Solve the adjoint system backward to get gradient"""

        # cut down on all the self calls
        dc = self.dc

        # initialize the adjoint momentum
        dc.Pa = 0

        # initialize the adjoint image (which contains residuals)
        m = self._m.residual(dc.J[-1], dc.Ifr[-1])
        jdet = la.det(vcalc.jacobian(dc.uf[-1], dc.curr_vox))
        mtil = jdet * self._t.apply_transform(m, dc.full_vox, dc.uf[-1])
        dc.Ia = mtil

        # initialize adjoint velocity
        m = self._t.resample(m, dc.full_vox, dc.curr_res)
        dI = vcalc.gradient(dc.I[-1], dc.curr_vox)
        va = - self._r.regularize(dI * m[..., np.newaxis])

        idx = -2
        i = dc.params['h'] - 2
        while i > -1:

            # advance the adjoint momentum
            dIva = np.einsum('...i,...i', dI, va)
            # forward Euler
            dc.Pa -= ((dc.t[i+1] - dc.t[i]) *
                      self._t.apply_transform(
                      dIva, dc.curr_vox, dc.uf[i+1]))

            # advance the adjoint image
            Pv = va * dc.P[i+1][..., np.newaxis]
            divPv = vcalc.divergence(Pv, dc.curr_vox)
            jdet = la.det(vcalc.jacobian(dc.uf[i+1], dc.curr_vox))
            # Forward Euler
            dc.Ia += ((dc.t[i+1] - dc.t[i]) * jdet *
                      self._t.apply_transform(
                      divPv, dc.curr_vox, dc.uf[i+1]))

            # Check if time point lines up with an image in the time series
            if dc.t[i] in dc.T:
                # add in the resulting residual
                m = self._m.residual(dc.J[idx], dc.Ifr[idx])
                jdet = la.det(vcalc.jacobian(dc.uf[i], dc.curr_vox))
                m = jdet * self._t.apply_transform(m, dc.full_vox, dc.uf[i])
                dc.Ia += m
                idx -= 1

            # advance adjoint velocity
            Phat = self._t.apply_transform(dc.Pa, dc.curr_vox, dc.ub[i])
            dPhat = vcalc.gradient(Phat, dc.curr_vox)
            A = dPhat * dc.P[i][..., np.newaxis]

            jdet = la.det(vcalc.jacobian(dc.ub[i], dc.curr_vox))
            Ihat = jdet * self._t.apply_transform(dc.Ia,
                                                  dc.curr_vox,
                                                  dc.ub[i])
            dI = vcalc.gradient(dc.I[i], dc.curr_vox)
            B = dI * Ihat[..., np.newaxis]

            va = self._r.regularize(A - B)

            i -= 1

        # return the gradient
        A = self._r.regularize(dI * dc.P[0][..., np.newaxis])
        A = dc.params['rat'] * np.einsum('...i,...i', dI, A)
        return A - dc.Pa

    def resample(self, res):
        """Resample all objects for multi-resolution schemes"""

        dc = self.dc
        if dc.params['iStep'] == 0.0:
                dc.Ifr[0] = np.copy(dc.J[0])
        dc.resample(res, self._t)
        self._r = regularizer.regularizer(dc.params['rType'],
                                          dc.params['a'],
                                          dc.params['b'],
                                          dc.params['c'],
                                          dc.params['d'],
                                          dc.curr_vox, res)


class data_container:
    """A container to wrap all data objects for geodesic regression"""

    def __init__(self, J, T, params):
        self.J = J
        self.T = T
        self.params = params

        self.N = J.shape[0]
        self.full_res = J[0].shape
        self.curr_res = J[0].shape
        self.full_vox = params['vox']
        self.curr_vox = params['vox']
        self.d = len(self.full_res)
        self.cfl_nums = np.zeros(params['h'])
        self.t = self.compute_t(self.T, self.params['h'])

        self.Ifr = np.empty_like(J)
        self.Ifr[0] = np.copy(J[0])
        self.txm = np.empty(self.full_res + (self.d,))

        self.P = np.zeros((1,) + self.full_res)
        self.I = np.reshape(np.copy(J[0]), (1,) + self.full_res)

    def resample(self, res, _t):
        """Change the resolution of the objects; ensure CFL is satisfied"""

        res_fact = max([float(res[i])/self.curr_res[i] for i in range(self.d)])
        self.cfl_nums *= res_fact
        ts = np.ceil(self.cfl_nums[:-1].max()) + 1
        if self.params['h'] <= ts:
            self.params['h'] = int(ts + 2)
            diff = self.params['h'] - len(self.cfl_nums)
            self.cfl_nums = np.pad(self.cfl_nums, (0, diff), mode='constant')
            self.t = self.compute_t(self.T, self.params['h'])

        P0 = self.P[0]
        self.P = np.empty((self.params['h'],) + res)
        self.P[0] = _t.resample(P0, self.curr_vox, res)
        self.I = np.empty_like(self.P)
        self.I[0] = _t.resample(self.Ifr[0], self.full_vox, res)

        self.curr_res = res
        self.curr_vox = _t.new_vox_size(self.full_res,
                                        res,
                                        self.full_vox)

        d = len(res)
        X = np.empty(res + (d,))
        sha = np.diag(res) - np.identity(d) + 1
        oa = np.ones(res)
        for i in range(d):
            X[..., i] = np.reshape(np.arange(res[i]), sha[i]) * (
                                            oa * self.curr_vox[i])

        self.uf = np.empty(self.P.shape + (self.d,))
        self.uf[0] = np.copy(X)
        self.ub = np.empty_like(self.uf)
        self.ub[0] = np.copy(X)

    def satisfy_cfl(self):
        """Ensure there are sufficient time steps to satisfy CFL condition"""

        ts = np.ceil(self.cfl_nums[:-1].max()) + 1
        if self.params['h'] <= ts:
            self.params['h'] = int(ts + 2)
            diff = self.params['h'] - len(self.cfl_nums)
            self.cfl_nums = np.pad(self.cfl_nums, (0, diff), mode='constant')
            self.t = self.compute_t(self.T, self.params['h'])

            pad_array = [(0, diff)] + [(0, 0)]*self.d
            self.P = np.pad(self.P, pad_array, mode='constant')
            self.I = np.pad(self.I, pad_array, mode='constant')

            pad_array += [(0, 0)]
            self.uf = np.pad(self.uf, pad_array, mode='constant')
            self.ub = np.pad(self.ub, pad_array, mode='constant')

    def compute_t(self, T, h):
        """Compute time points along discrete sampling of geodesic"""

        t = (T[-1] - T[0])/(h - 1.) * np.arange(h) + T[0]
        # Shift subset of points in t to equal sampling times
        for time in T:
            idx = min(range(h), key=lambda i: abs(t[i] - time))
            t[idx] = time
        return t

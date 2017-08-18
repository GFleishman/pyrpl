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


class sigsid:
    """Symmetric Interleaved Geodesic Shooting in Diffeomorphisms 
    nonlinear registration model

    This model fits a geodesic of diffeomorphisms, parameterized by a scalar
    initial momentum field, through the given time series of images. This
    symmetric interleaved version alternates shooting from either side
    of the geodesic to render the result robust to switching the order
    of the inputs; i.e. symmetric up to numerical accuracy."""

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


class data_container:
    """A container to wrap all data objects for geodesic regression"""

    def __init__(self, J, T, params):
        
        # just to cut down on ugly text
        s = self

        # store references to the inputs
        s.J = J
        s.T = T
        s.params = params

        # store references to the current and full resolution
        s.full_res = J[0].shape
        s.curr_res = J[0].shape
        s.full_vox = params['vox']
        s.curr_vox = params['vox']
        # store reference to the dimension
        s.d = len(s.full_res)

        # allocate array to store cfl number for each time step
        s.cfl_nums = np.zeros(params['h'])
        # compute the time step values
        s.t = s.compute_times(T, params['h'])

        # allocate array for full resolution endpoints of image path
        s.Ifr = np.empty_like(J)
        np.copyto(s.Ifr[0], J[0])
        # and array for upsampling transformation
        s.txm = np.empty(s.full_res + (s.d,))

        # allocate arrays to store momenta and image paths
        s.P = np.zeros((1,) + s.full_res)
        s.I = np.empty((1,) + s.full_res)
        np.copyto(s.I[0], J[0])

    def resample(self, res, _t):
        """Change the resolution of the objects; ensure CFL is satisfied"""

        # just to cut down on ugly text
        s = self

        # update cfl numbers to accomodate resolution change
        res_fact = max([float(res[i])/s.curr_res[i] for i in range(s.d)])
        s.cfl_nums *= res_fact
        s.check_cfl()

        # reallocate momenta and image paths at new res
        P0 = s.P[0]
        s.P = np.empty((s.params['h'],) + res)
        s.P[0] = _t.resample(P0, s.curr_vox, res)
        s.I = np.empty_like(s.P)
        s.I[0] = _t.resample(s.Ifr[0], s.full_vox, res)

        # update resolution and voxel references
        s.curr_res = res
        s.curr_vox = _t.new_vox_size(s.full_res, res, s.full_vox)

        # reallocate forward and backward position arrays at new res
        X = _t.position_array(res, s.curr_vox)
        s.uf = np.empty(s.P.shape + (s.d,))
        np.copyto(s.uf[0], X)
        s.ub = np.empty_like(s.uf)
        np.copyto(s.ub[0], X)

    def check_cfl(self):
        """get max time steps by cfl adjust cfl and time arrays accordingly"""

        # just to cut down on ugly text
        s = self

        ts = np.ceil(s.cfl_nums[:-1].max()) + 1
        if s.params['h'] <= ts:
            s.params['h'] = int(ts + 2)
            diff = s.params['h'] - len(s.cfl_nums)
            s.cfl_nums = np.pad(s.cfl_nums, (0, diff), mode='constant')
            s.t = s.compute_times(s.T, s.params['h'])
            return diff
        else:
            return False

    def satisfy_cfl(self):
        """Ensure there are sufficient time steps to satisfy CFL condition"""

        # just to cut down on ugly text
        s = self

        # if cfl needs to be updated
        diff = s.check_cfl()
        if diff:
            pad_array = [(0, diff)] + [(0, 0)]*s.d
            s.P = np.pad(s.P, pad_array, mode='constant')
            s.I = np.pad(s.I, pad_array, mode='constant')

            pad_array += [(0, 0)]
            s.uf = np.pad(s.uf, pad_array, mode='constant')
            s.ub = np.pad(s.ub, pad_array, mode='constant')

    def compute_times(self, T, h):
        """Compute time points along discrete sampling of geodesic"""

        t = (T[-1] - T[0])/(h - 1.) * np.arange(h) + T[0]
        # Shift subset of points in t to equal sampling times
        for time in T:
            idx = min(range(h), key=lambda i: abs(t[i] - time))
            t[idx] = time
        return t

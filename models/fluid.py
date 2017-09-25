# -*- coding: utf-8 -*-

import numpy as np
import pyrpl.image_tools.transformer as transformer
import pyrpl.image_tools.matcher as matcher
import pyrpl.image_tools.regularizer as regularizer


# TODO: update to actual fluid model!
class model:
    
    def __init__(self, input_dictionary):
        """Initialize image level tools"""

        # initialize all the model objects
        self.smooth_counter = 1
        self._t = transformer.transformer()
        self.dc = data_container(input_dictionary['image_data'],
                                 input_dictionary, self._t)
        self._m = matcher.matcher(input_dictionary['matcher'],
                                  input_dictionary['window'])
        self._r = regularizer.regularizer(input_dictionary['regularizer'],
                                          input_dictionary['abcd'][0],
                                          input_dictionary['abcd'][1],
                                          input_dictionary['abcd'][2],
                                          input_dictionary['abcd'][3],
                                          self.dc.curr_vox,
                                          self.dc.curr_res)

    def evaluate(self):
        """Evaluate the objective functional"""

        dc = self.dc
        if dc.curr_res != dc.full_res:
            phi_hr = self._t.resample(dc.phi+dc.u,
                                      dc.curr_vox, dc.full_res,
                                      vec=True)
            dc.I0 = self._t.apply_transform(dc.J1, dc.full_vox, phi_hr)
        else:
            dc.I0 = self._t.apply_transform(dc.J1, dc.full_vox, dc.phi+dc.u)
            
        # compute regularizer term
        u_energy = 0.001  # temporary, should be elastic potential energy of u
        # compute matching functional
        obj_func = [self._m.dist(dc.J0, dc.I0), self._m.dist(dc.J0, dc.I0)]
        return [u_energy] + [obj_func]

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
        if self.smooth_counter % 5 == 0:
            self.dc.u = self._r.regularize(self.dc.u)
        self.smooth_counter += 1

    def resample(self, res):
        """Resample all objects for multi-resolution schemes"""

        dc = self.dc
        dc.resample(res, self._t)
        self._r = regularizer.regularizer(dc.params['regularizer'],
                                          dc.params['abcd'][0],
                                          dc.params['abcd'][1],
                                          dc.params['abcd'][2],
                                          dc.params['abcd'][3],
                                          self.dc.curr_vox, res)
    
    def get_original_image(self, i):
        """Return the ith original image"""
        
        if i == 0:
            return self.dc.J0
        elif i == 1:
            return self.dc.J1
    
    def get_warped_image(self, i):
        """Return the ith warped image"""
        
        return self.dc.I0
    
    def get_warp(self, i):
        """Get warp of moving coords onto target coords"""
        
        return self.dc.phi + self.dc.u
    
    def get_current_voxel(self):
        """Get current voxel size"""
        
        return self.dc.curr_vox


class data_container:
    """A container for elastic registration data"""

    def __init__(self, J, params, _t):
        
        # just to cut down on ugly text
        s = self
        
        s.J0 = J[0]
        s.J1 = J[1]
        s.params = params

        s.I0 = np.copy(J[1])
        s.d = len(J[0].shape)
        s.full_res = J[0].shape
        s.curr_res = J[0].shape
        s.full_vox = params['voxel']
        s.curr_vox = params['voxel']

        s.u = np.zeros(tuple(s.curr_res) + (s.d,))
        s.phi = np.empty_like(s.u)
        np.copyto(s.phi, _t.position_array(s.curr_res, s.curr_vox))
        

    def resample(self, res, _t):

        # just to cut down on ugly text
        s = self

        s.u = _t.resample(s.u, s.curr_vox, res, vec=True)
        s.curr_res = res
        s.curr_vox = _t.new_vox_size(s.full_res, res, s.full_vox)
        s.phi = np.empty(tuple(s.curr_res) + (s.d,))
        np.copyto(s.phi, _t.position_array(s.curr_res, s.curr_vox))

# -*- coding: utf-8 -*-
"""
Author: Greg M. Fleishman

Description: Method for fitting a geodesic through an image time series

Dependencies: NumPy, MatPlotLib, and pyrt.regTools
"""

import numpy as np
import PyRPL.models.geodesic_regression_in_diffeomorphisms as model


def optimize(J, T, params):
    """Find geodesic parameters optimal for image time series J at times T"""

    # arrays to store objective function and gradient magnitude
    # values throughout optimization
    ttl_its = np.sum(params['its'])
    P0_mag = np.zeros(ttl_its)
    data_match = np.zeros((J.shape[0], ttl_its))
    grad_mag = np.zeros(ttl_its)
    stop = 1.

    # a geodesic_optimizer object, solves forward and backward systems
    gr = model.geodesic_regression_in_diffeomorphisms(J, T, params)

    k = 0   # count the total iterations
    j = 0   # count the resamples
    while k < np.sum(params['its']) and stop > 0.0002:

        # make sure CFL condition is satisfied
        gr.dc.satisfy_cfl()

        # resample when necessary
        if k == np.sum(params['its'][0:j]):
            gr.resample(params['res'][j])

            j += 1

        # evaluate objective func, get gradient, take descent step
        obj_func = gr.evaluate()
        g = gr.get_gradient()
        gr.take_step(- params['pStep'] * g)

        # store objective function values
        P0_mag[k] = obj_func[0]
        for i in range(J.shape[0]):
            data_match[i, k] = obj_func[i+1]

        # compute and store gradient magnitude
        kg = gr._r.regularize(g[..., np.newaxis]).squeeze()
        grad_mag[k] = np.prod(gr.dc.curr_vox) * np.sum(g * kg)

        # update stopping criteria
        stop = grad_mag[k]/grad_mag[0]
        k += 1

    return gr.dc, P0_mag, data_match, grad_mag

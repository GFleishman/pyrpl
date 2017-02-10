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

    # a list to store step sizes
    steps = []

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

        # evaluate objective func, get gradient
        obj_func = gr.evaluate()
        g = gr.get_gradient()

        # compute and store gradient magnitude
        sd = np.copy(g)
        ksd = gr._r.regularize(np.copy(sd)[..., np.newaxis]).squeeze()
        grad_mag[k] = np.prod(gr.dc.curr_vox) * np.sum(sd * ksd)

        # store objective function values
        P0_mag[k] = obj_func[0]
        for i in range(gr.dc.N):
            data_match[i, k] = obj_func[i+1]

        # take the first secant method step
        step = params['pStep']
        gr.take_step(- step * sd)
        local_steps = [step]

        # iterate the secant method line search (max 4 times)
        old_ip = grad_mag[k] / np.prod(gr.dc.curr_vox)
        for sec_step in range(4):
            # compute the gradient at the new position
            gr.evaluate()
            g = gr.get_gradient()
            new_ip = np.sum(g * ksd)
            step *= -new_ip / (new_ip - old_ip)
            local_steps.append(step)
            gr.take_step(-step * sd)
            old_ip = new_ip
            if step**2 * grad_mag[k] <= 1e-5:
                break

        # update stopping criteria
        steps.append(local_steps)
        stop = grad_mag[k]/grad_mag[0]
        k += 1

    return P0_mag, data_match, grad_mag, steps

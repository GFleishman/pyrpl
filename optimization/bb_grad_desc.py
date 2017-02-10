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

    # a flag to help reduce redundant forward solves
    resolve = True

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
        if resolve:
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

        # BB step size
        if k == 0 or k == gr.dc.params['its'][0]:
            old_grad = g
            old_step = gr.dc.params['pStep']
            local_steps = [old_step]
            gr.dc.P[0] -= old_step * old_grad
        else:
            g_diff = g - old_grad
            gd_smooth = gr._r.regularize(g_diff[..., np.newaxis]).squeeze()
            step = - np.sum(gd_smooth * old_grad) * old_step
            step *= 1.0/np.sum(gd_smooth * g_diff)
            local_steps = [step]
            gr.dc.P[0] -= step * g
            old_grad = g
            old_step = step

            # non-monotonic backtracking line search (max 10 backsteps)
            resolve = True
            M = max(0, k - 10)
            thresh = max(-data_match[1, M:k] + P0_mag[M:k])
            for back_step in range(10):
                obj_func = gr.solveForward(gr.dc)
                curr_obj_val = obj_func[0] - obj_func[-1]
                if curr_obj_val > thresh - 1e-4 * step * grad_mag[k]:
                    old_step *= 0.5
                    local_steps.append(step)
                    gr.dc.P[0] += old_step * g
                else:
                    resolve = False
                    break

        # update stopping criteria
        steps.append(local_steps)
        stop = grad_mag[k]/grad_mag[0]
        k += 1

    return P0_mag, data_match, grad_mag, steps

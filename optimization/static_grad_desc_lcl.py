# -*- coding: utf-8 -*-
"""
Author: Greg M. Fleishman

Description: Method for fitting a geodesic through an image time series

Dependencies: NumPy, MatPlotLib, and pyrt.regTools
"""

# fundamental numerical imports
import numpy as np
import numpy.linalg as la

# fundamental image level imports
import PyRPL.image_tools.vcalc as vcalc
import PyRPL.models.elastic_registration as model

# visualization imports
import matplotlib.pyplot as plt
from matplotlib import cm


def optimize(J, T, params):
    """Find geodesic parameters optimal for image time series J at times T"""

    # arrays to store objective function and gradient magnitude
    # values throughout optimization
    P0_mag = []
    data_match = []
    grad_mag = []
    stop = 1.

    # a geodesic_optimizer object, solves forward and backward systems
    gr = model.elastic_registration(J, T, params)

    k = 0   # count the total iterations
    j = 0   # count the resamples
    while k < np.sum(params['its']) and stop > 0.0002:

        # resample when necessary
        if k == np.sum(params['its'][0:j]):
            gr.resample(params['res'][j])

            j += 1

        # evaluate objective func, get gradient, take descent step
        obj_func = gr.evaluate()
        g = gr.get_gradient()
        gr.take_step(- params['pStep'] * g[0])

        # store objective function values
        P0_mag.append(obj_func[0])
        data_match.append(obj_func[1:])

        # compute and store gradient magnitude
        grad_mag.append(g[1])

        # update stopping criteria
        stop = grad_mag[k]/grad_mag[0]
        k += 1

        # display feedback for testing
        display_feedback_elastic(P0_mag, data_match, k, gr)

    return gr.dc, P0_mag, data_match, grad_mag


def display_feedback(P0_mag, data_match, k, gr):
    # 2D feedback windows
    fig = plt.figure('registerer level feedback', figsize=(16, 10))
    plt.clf()
    for n in range(gr.dc.N):
        # original image time series
        fig.add_subplot(4, gr.dc.N, n+1)
        plt.imshow(np.rot90(gr.dc.J[n]), cmap=cm.gray)
        plt.axis('off')
        # geodesic images
        fig.add_subplot(4, gr.dc.N, gr.dc.N+n+1)
        plt.imshow(np.rot90(gr.dc.Ifr[n]), cmap=cm.gray)
        plt.axis('off')
        # jacobian determinants
        idx = np.where(gr.dc.t == gr.dc.T[n])[0][0]
        jd = la.det(vcalc.jacobian(gr.dc.uf[idx], gr.dc.curr_vox))
        jd = np.log10(jd)
        fig.add_subplot(4, gr.dc.N, 2*gr.dc.N+n+1)
        plt.imshow(np.rot90(jd))
        plt.axis('off')
        plt.colorbar()

    fig.add_subplot(4, gr.dc.N, 2*gr.dc.N + 1)
    plt.cla()
    plt.imshow(np.rot90(gr.dc.P[0]), cmap=cm.gray)
    plt.axis('off')
    plt.colorbar()
    fig.add_subplot(4, gr.dc.N, 3*gr.dc.N + 1)
    for i in range(1, gr.dc.N):
        plt.plot(np.arange(k), data_match[i, :k])
    fig.add_subplot(4, gr.dc.N, 3*gr.dc.N + 2)
    plt.plot(np.arange(k), P0_mag[:k])

    plt.pause(0.001)
    plt.draw()


def display_feedback_elastic(P0_mag, data_match, k, gr):
    # 2D feedback windows
    fig = plt.figure('registerer level feedback', figsize=(16, 10))
    plt.clf()
    gr.dc.N = 2
    #
    fig.add_subplot(3, 3, 1)
    plt.imshow(np.rot90(gr.dc.J0), cmap=cm.gray)
    plt.axis('off')
    #
    fig.add_subplot(3, 3, 2)
    plt.imshow(np.rot90(gr.dc.J1), cmap=cm.gray)
    plt.axis('off')
    #
    fig.add_subplot(3, 3, 4)
    plt.imshow(np.rot90(gr.dc.I0), cmap=cm.gray)
    plt.axis('off')
    #
    fig.add_subplot(3, 3, 5)
    plt.imshow(np.rot90(gr.dc.J1), cmap=cm.gray)
    plt.axis('off')
    #
    fig.add_subplot(3, 3, 7)
    l = [x[0] for x in data_match]
    plt.plot(l)
    #
    fig.add_subplot(3, 3, 8)
    plt.plot(P0_mag)
    #
    fig.add_subplot(3, 3, 9)
    jd = la.det(vcalc.jacobian(gr.dc.u, gr.dc.curr_vox))
    plt.imshow(np.rot90(jd))
    plt.axis('off')
    plt.colorbar()
    #
    plt.pause(0.001)
    plt.draw()

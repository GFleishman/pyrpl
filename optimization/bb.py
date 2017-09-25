# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
import pyrpl.image_tools.vcalc as vcalc
import pyrpl.models.modeler as modeler


# TODO: include steps list in output
def optimize(input_dictionary):
    """Find geodesic parameters optimal for image time series J at times T"""

    # initialize
    steps = []          # store the calculated step sizes
    regu_match = []     # store regularization term from objective func
    data_match = []     # store data match term
    grad_mag = []       # store gradient magnitude
    stop = 1.
    model = modeler.model(input_dictionary)     # the actual model

    k = 0   # count the total iterations
    j = 0   # count the resamples
    resolve = True  # a flag to reduce number of model evaluations
    # TODO: make "stop" available to user as parameter
    while k < np.sum(input_dictionary['iterations']) and stop > 0.0002:
        # resample when necessary
        if k == np.sum(input_dictionary['iterations'][0:j]):
            model.resample(input_dictionary['resolutions'][j])
            j += 1

        # evaluate objective func, get gradient
        if resolve:
            obj_func = model.evaluate()
        grad = model.get_gradient()

        # store optimization values
        regu_match.append(obj_func[0])
        data_match.append(obj_func[1])
        grad_mag.append(grad[1])

        # BB step size
        if k == 0 or k in input_dictionary['iterations']:
            old_grad = grad[0]
            old_step = input_dictionary['step']
            local_steps = [old_step]
            model.take_step(-old_step * old_grad)
        else:
            # convolution is a linear operator. I may be able to rewrite
            # the math here to be faster or more efficient
            g_diff = grad[0] - old_grad
            gd_smooth = model._r.regularize(g_diff[..., np.newaxis]).squeeze()
            step = - np.sum(gd_smooth * old_grad) * old_step
            step *= 1.0/np.sum(gd_smooth * g_diff)
            local_steps = [step]
            model.take_step(-step * grad[0])
            old_grad = grad[0]
            old_step = step

            # non-monotonic backtracking line search (max 10 backsteps)
            resolve = True
            M = max(0, k - 10)
            thresh = max([-sum(x) for x in data_match[M:k]] + regu_match[M:k])
            for back_step in range(10):
                obj_func = model.evaluate()
                curr_obj_val = obj_func[0] - sum(obj_func[1])
                if curr_obj_val > thresh - 1e-4 * step * grad_mag[k]:
                    old_step *= 0.5
                    local_steps.append(step)
                    model.take_step(old_step * grad[0])
                else:
                    resolve = False
                    break

        # update stopping criteria
        steps.append(local_steps)
        stop = grad_mag[k]/grad_mag[0]
        k += 1

        # display feedback for testing
        if input_dictionary['feedback']:
            display_feedback(regu_match, data_match, k, model)


    output = model.package_output()
    output['objective'] = [regu_match, data_match, grad_mag]
    return output


# display a real time feedback window
def display_feedback(regu_match, data_match, k, model):
    """Display feedback window with objective function values and images"""
    # 2D feedback windows
    fig = plt.figure('Optimization', figsize=(16, 10))
    plt.clf()
    for i in range(2):
        # original input images
        fig.add_subplot(2, 3, i+1)
        plt.imshow(np.rot90(model.dc.J[i]), cmap=cm.gray)
        plt.axis('off')
        # template and match to target
        fig.add_subplot(2, 3, i+4)
        plt.imshow(np.rot90(model.dc.Ifr[i]), cmap=cm.gray)
        plt.axis('off')
    # objective function values
    fig.add_subplot(2, 3, 3)
    for i in range(2):
        dm = [x[i] for x in data_match]
        plt.plot(range(k), dm)  # TODO: add color
    fig.add_subplot(2, 3, 3)
    plt.plot(range(k), regu_match)
    # jacobian determinants
    fig.add_subplot(2, 3, 6)
    jd = la.det(vcalc.jacobian(model.get_warp(), model.get_current_vox()))
    jd = np.log10(jd)
    plt.imshow(np.rot90(jd))
    plt.axis('off')
    plt.colorbar()
    # show plot
    plt.pause(0.001)
    plt.draw()

# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
import pyrpl.image_tools.vcalc as vcalc
import pyrpl.models.modeler as modeler


# gradient descent with static step size
def optimize(input_dictionary):
    """Optimize objective function with static gradient descent step size"""

    # initialize
    regu_match = []     # store regularization term from objective func
    data_match = []     # store data match term
    grad_mag = []       # store gradient magnitude
    stop = 1.
    model = modeler.model(input_dictionary)     # the actual model

    k = 0   # count the total iterations
    j = 0   # count the resamples
    # TODO: make "stop" available to user as parameter
    while k < np.sum(input_dictionary['iterations']): # and stop > 0.0002:
        # resample when necessary
        if k == np.sum(input_dictionary['iterations'][0:j]):
            model.resample(input_dictionary['resolutions'][j])
            j += 1

        # the actual gradient descent
        obj_func = model.evaluate()
        grad = model.get_gradient()
        model.take_step(- input_dictionary['step'] * grad[0])

        # store optimization values
        regu_match.append(obj_func[0])
        data_match.append(obj_func[1])
        grad_mag.append(grad[1])

        # update stopping criteria
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
        plt.imshow(np.rot90(model.get_original_image(i)), cmap=cm.gray)
        plt.axis('off')
        # template and match to target
        fig.add_subplot(2, 3, i+4)
        plt.imshow(np.rot90(model.get_warped_image(i)), cmap=cm.gray)
        plt.axis('off')
    # objective function values
    fig.add_subplot(2, 3, 3)
    colors = ['blue', 'green', 'red']
    for i in range(2):
        dm = [x[i] for x in data_match]
        plt.plot(range(k), dm, color=colors[i])
    fig.add_subplot(2, 3, 3)
#    plt.plot(range(k), regu_match, color=colors[-1])
    # jacobian determinants
    fig.add_subplot(2, 3, 6)
    jd = la.det(vcalc.jacobian(model.get_warp(-1), model.get_current_voxel()))
    jd = np.log10(jd)
    plt.imshow(np.rot90(jd))
    plt.axis('off')
    plt.colorbar()
    # show plot
    plt.pause(0.001)
    plt.draw()

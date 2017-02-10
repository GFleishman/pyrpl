"""
Author: Greg M. Fleishman

Description: A test of the accuracy of vcalc functions by comparing them
to analytical values for a function

Dependencies: NumPy and MatPlotLib, pyrt.regTools.vcalc
"""

import sys
import numpy as np
from matplotlib import pyplot as plt
import vcalc

# Establish constants
sn = np.sin
cs = np.cos
R = 2*np.pi
res = np.array([128, 128])
vox = np.array([1/128.0, 1/128.0])
rngx = np.arange(0, res[0])*vox[0]
rngy = np.arange(0, res[1])*vox[1]
x, y = np.meshgrid(rngx, rngy, indexing='ij')


def create_test_img():
    """Create test img/function"""
    return sn(R*x) + cs(R*y) - sn(R*y)*cs(R*x)


def compute_analytical_derivatives(img):
    """Use formula to calculate analytical derivatives"""
    # gradient
    grad = np.empty(img.shape + (2,))
    grad[..., 0] = cs(R*x)*R + sn(R*y)*sn(R*x)*R
    grad[..., 1] = -sn(R*y)*R - cs(R*y)*cs(R*x)*R

    # jacobian of gradient
    jac = np.empty(img.shape + (2, 2))
    jac[..., 0, 0] = -sn(R*x)*(R)**2 + sn(R*y)*cs(R*x)*(R)**2
    jac[..., 0, 1] = cs(R*y)*sn(R*x)*(R)**2
    jac[..., 1, 0] = cs(R*y)*sn(R*x)*(R)**2
    jac[..., 1, 1] = -cs(R*y)*(R)**2 + sn(R*y)*cs(R*x)*(R)**2

    # jacobian determinant
    jd = (jac[..., 0, 0]*jac[..., 1, 1] - jac[..., 0, 1]*jac[..., 1, 0])

    # divergence of gradient
    div = jac[..., 0, 0] + jac[..., 1, 1]

    return (grad, jac, jd, div)


def compute_numerical_derivatives(img):
    """Use vcalc to compute numerical values of derivatives"""
    # gradient
    grad = vcalc.gradient(img, vox)

    # jacobian of gradient
    jac = vcalc.jacobian(grad, vox, txm=False)

    # jacobian determinant
    jd = np.linalg.det(jac)

    # divergence of gradient
    div = vcalc.divergence(grad, vox)

    return (grad, jac, jd, div)


def visualize_results(img, ana, num):
    """visulize results"""

    # fetch results
    ana_grad = ana[0]
    ana_jac = ana[1]
    ana_jd = ana[2]
    ana_div = ana[3]

    num_grad = num[0]
    num_jac = num[1]
    num_jd = num[2]
    num_div = num[3]

    # gradient
    fig = plt.figure('gradient')

    mn = np.max([ana_grad[..., 0].min(), num_grad[..., 0].min()])
    mx = np.min([ana_grad[..., 0].max(), num_grad[..., 0].max()])
    fig.add_subplot(2, 2, 1)
    plt.imshow(ana_grad[..., 0], vmin=mn, vmax=mx)
    plt.axis('off')
    fig.add_subplot(2, 2, 2)
    plt.imshow(num_grad[..., 0], vmin=mn, vmax=mx)
    plt.axis('off')
    plt.colorbar()

    mn = np.max([ana_grad[..., 1].min(), num_grad[..., 1].min()])
    mx = np.min([ana_grad[..., 1].max(), num_grad[..., 1].max()])
    fig.add_subplot(2, 2, 3)
    plt.imshow(ana_grad[..., 1], vmin=mn, vmax=mx)
    plt.axis('off')
    fig.add_subplot(2, 2, 4)
    plt.imshow(num_grad[..., 1], vmin=mn, vmax=mx)
    plt.axis('off')
    plt.colorbar()

    # jacobian
    fig = plt.figure('jacobian')

    mn = np.max([ana_jac[..., 0, 0].min(), num_jac[..., 0, 0].min()])
    mx = np.min([ana_jac[..., 0, 0].max(), num_jac[..., 0, 0].max()])
    fig.add_subplot(4, 2, 1)
    plt.imshow(ana_jac[..., 0, 0], vmin=mn, vmax=mx)
    plt.axis('off')
    fig.add_subplot(4, 2, 2)
    plt.imshow(num_jac[..., 0, 0], vmin=mn, vmax=mx)
    plt.axis('off')
    plt.colorbar()

    mn = np.max([ana_jac[..., 0, 1].min(), num_jac[..., 0, 1].min()])
    mx = np.min([ana_jac[..., 0, 1].max(), num_jac[..., 0, 1].max()])
    fig.add_subplot(4, 2, 3)
    plt.imshow(ana_jac[..., 0, 1], vmin=mn, vmax=mx)
    plt.axis('off')
    fig.add_subplot(4, 2, 4)
    plt.imshow(num_jac[..., 0, 1], vmin=mn, vmax=mx)
    plt.axis('off')
    plt.colorbar()

    mn = np.max([ana_jac[..., 1, 0].min(), num_jac[..., 1, 0].min()])
    mx = np.min([ana_jac[..., 1, 0].max(), num_jac[..., 1, 0].max()])
    fig.add_subplot(4, 2, 5)
    plt.imshow(ana_jac[..., 1, 0], vmin=mn, vmax=mx)
    plt.axis('off')
    fig.add_subplot(4, 2, 6)
    plt.imshow(num_jac[..., 1, 0], vmin=mn, vmax=mx)
    plt.axis('off')
    plt.colorbar()

    mn = np.max([ana_jac[..., 1, 1].min(), num_jac[..., 1, 1].min()])
    mx = np.min([ana_jac[..., 1, 1].max(), num_jac[..., 1, 1].max()])
    fig.add_subplot(4, 2, 7)
    plt.imshow(ana_jac[..., 1, 1], vmin=mn, vmax=mx)
    plt.axis('off')
    fig.add_subplot(4, 2, 8)
    plt.imshow(num_jac[..., 1, 1], vmin=mn, vmax=mx)
    plt.axis('off')
    plt.colorbar()

    # divergence
    fig = plt.figure('divergence')

    mn = np.max([ana_div.min(), num_div.min()])
    mx = np.min([ana_div.max(), num_div.max()])
    fig.add_subplot(1, 2, 1)
    plt.imshow(ana_div, vmin=mn, vmax=mx)
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(num_div, vmin=mn, vmax=mx)
    plt.axis('off')
    plt.colorbar()

    # jacobian determinant
    fig = plt.figure('jacobian determinant')

    mn = np.max([ana_jd.min(), num_jd.min()])
    mx = np.min([ana_jd.max(), num_jd.max()])
    fig.add_subplot(1, 2, 1)
    plt.imshow(ana_jd, vmin=mn, vmax=mx)
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(num_jd, vmin=mn, vmax=mx)
    plt.axis('off')
    plt.colorbar()

    # the original image
    plt.figure('original image')
    plt.imshow(img)
    plt.axis('off')
    plt.colorbar()

    plt.show()


def main():
    """main entry point for execution"""
    img = create_test_img()
    ana = compute_analytical_derivatives(img)
    num = compute_numerical_derivatives(img)
    visualize_results(img, ana, num)


if __name__ == '__main__':
    sys.exit(main())

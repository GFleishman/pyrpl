"""
Author: Greg M. Fleishman

Description: Test the transformer accuracy against an analytical calculation

Dependencies: NumPy, MatPlotLib, and pyrt.regTools.transformer
"""

import sys
import numpy as np
import transformer
from matplotlib import pyplot as plt


# Establish constants
resx = 128
resy = 128
vox = np.array([0.011, 0.009])
rngx = np.arange(0, resx)*vox[0]
rngy = np.arange(0, resy)*vox[1]
x, y = np.meshgrid(rngx, rngy, indexing='ij')
R = 2*np.pi
sn = np.sin
cs = np.cos

nresx = 64
nresy = 256
nvox = []
nvox.append(vox[0]*resx/float(nresx))
nvox.append(vox[1]*resy/float(nresy))

ttlx = cs(R*x)*sn(R*y)/15.0
ttly = sn(R*x)*cs(R*y)/15.0


def create_test_img():
    """Create test image"""
    return sn(R*x) + cs(R*y) - sn(R*y)*cs(R*x)


def resample_analytical(img):
    """Analytically resample image"""
    nrngx = np.arange(0, nresx)*nvox[0]
    nrngy = np.arange(0, nresy)*nvox[1]
    nx, ny = np.meshgrid(nrngx, nrngy, indexing='ij')

    return sn(R*nx) + cs(R*ny) - sn(R*ny)*cs(R*nx)


def resample_numerical(img, _t):
    """numerically resample image"""
    return _t.regrid(img, vox, (nresx, nresy))


def transform_analytical(img):
    """Analytically transform image"""
    xt = x + ttlx
    yt = y + ttly

    xt = xt % (resx*vox[0])
    yt = yt % (resy*vox[1])

    return sn(R*xt) + cs(R*yt) - sn(R*yt)*cs(R*xt)


def transform_numerical(img, _t):
    """numerically transform image"""
    _t._initialize(img.shape)
    txm = np.empty(img.shape + (2,))
    txm[..., 0] = ttlx
    txm[..., 1] = ttly
    return _t.applyTransform(img, vox, txm)


def visualize_results(img, ares_img, nres_img, atxm_img, ntxm_img):
    """visualize the results for accuracy"""
    fig = plt.figure('resamples', figsize=(12, 8))
    fig.add_subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.colorbar()
    fig.add_subplot(1, 3, 2)
    plt.imshow(ares_img)
    plt.axis('off')
    plt.colorbar()
    fig.add_subplot(1, 3, 3)
    plt.imshow(nres_img)
    plt.axis('off')
    plt.colorbar()

    fig = plt.figure('Transforms', figsize=(16, 10))
    fig.add_subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.colorbar()
    fig.add_subplot(1, 3, 2)
    plt.imshow(atxm_img)
    plt.axis('off')
    plt.colorbar()
    fig.add_subplot(1, 3, 3)
    plt.imshow(ntxm_img)
    plt.axis('off')
    plt.colorbar()

    plt.show()


def main():
    _t = transformer.transformer()
    img = create_test_img()
    ares_img = resample_analytical(img)
    nres_img = resample_numerical(img, _t)
    atxm_img = transform_analytical(img)
    ntxm_img = transform_numerical(img, _t)

    print "voxel resize correct: " + str(np.allclose(nres_img[1], nvox))
    visualize_results(img, ares_img, nres_img[0], atxm_img, ntxm_img)


if __name__ == '__main__':
    sys.exit(main())

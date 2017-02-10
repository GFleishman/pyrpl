# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:56:36 2015

@author: gfleishman
"""

import sys
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm

pid = sys.argv[1]
wPath = sys.argv[2]
p = sys.argv[3]

paths = glob.glob(p+'/*/la/'+pid+'/*.nii.gz')
paths.sort()
n = len(paths)

mask_path = p + '/masks/' + pid + '/' + pid + '_sc_mask.nii.gz'
mask = nib.load(mask_path).get_data().squeeze()
rgbmask = np.empty(mask.shape + (3,))
rgbmask[:, :, :, 0] = mask

fig = plt.figure('Final QC Image', figsize=(14, 9))

for p in range(n):

    plt.figtext(x=0.05, y=0.98-0.02*p, s=paths[p])
    img = nib.load(paths[p]).get_data().squeeze()

    fig.add_subplot(n, 3, 3*p+1)
    plt.imshow(np.rot90(img[110, :, :]), cmap=cm.gray)
    plt.imshow(np.rot90(rgbmask[110, :, :]), alpha=0.25)
    plt.axis('off')

    fig.add_subplot(n, 3, 3*p+2)
    plt.imshow(np.rot90(img[:, 110, :]), cmap=cm.gray)
    plt.imshow(np.rot90(rgbmask[:, 110, :]), alpha=0.25)
    plt.axis('off')

    fig.add_subplot(n, 3, 3*p+3)
    plt.imshow(np.rot90(img[:, :, 110]), cmap=cm.gray)
    plt.imshow(np.rot90(rgbmask[:, :, 110]), alpha=0.25)
    plt.axis('off')

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(wPath)

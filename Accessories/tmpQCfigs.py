# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:56:36 2015

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm

imgPath = sys.argv[1]
wPath = sys.argv[2]

img = nib.load(imgPath).get_data().squeeze()
fig = plt.figure('tmp QC Image', figsize=(14, 9))

for i in range(3):
    fig.add_subplot(3, 3, 1+i)
    plt.imshow(np.rot90(img[80+i*30, :, :]), cmap=cm.gray)
    plt.axis('off')

for i in range(3):
    fig.add_subplot(3, 3, 4+i)
    plt.imshow(np.rot90(img[:, 80+i*30, :]), cmap=cm.gray)
    plt.axis('off')

for i in range(3):
    fig.add_subplot(3, 3, 7+i)
    plt.imshow(np.rot90(img[:, :, 80+i*30]), cmap=cm.gray)
    plt.axis('off')

plt.subplots_adjust(wspace=None, hspace=None)
plt.savefig(wPath)

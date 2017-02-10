# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:43:00 2015

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm

img = sys.argv[1]
msk = sys.argv[2]
savePath = sys.argv[3]

img = nib.load(img).get_data().squeeze()
msk = nib.load(msk).get_data().squeeze()

rgbmsk = np.empty(msk.shape + (3,))
rgbmsk[:, :, :, 0] = msk

fig = plt.figure('QC Image', figsize=(14, 9))
plt.figtext(x=0.452, y=0.96, s=str(img.shape))
plt.figtext(x=0.452, y=0.92, s=str(msk.shape))

for i in range(3):
    fig.add_subplot(3, 3, 1+i)
    plt.imshow(np.rot90(img[108+i*20, :, :]), cmap=cm.gray)
    plt.imshow(np.rot90(rgbmsk[108+i*20, :, :]), alpha=0.25)
    plt.axis('off')

for i in range(3):
    fig.add_subplot(3, 3, 4+i)
    plt.imshow(np.rot90(img[:, 108+i*20, :]), cmap=cm.gray)
    plt.imshow(np.rot90(rgbmsk[:, 108+i*20, :]), alpha=0.25)
    plt.axis('off')

for i in range(3):
    fig.add_subplot(3, 3, 7+i)
    plt.imshow(np.rot90(img[:, :, 63+i*20]), cmap=cm.gray)
    plt.imshow(np.rot90(rgbmsk[:, :, 63+i*20]), alpha=0.25)
    plt.axis('off')

plt.subplots_adjust(wspace=None, hspace=None)
plt.savefig(savePath)

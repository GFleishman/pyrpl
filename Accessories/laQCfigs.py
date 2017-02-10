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

pid = sys.argv[1]
wPath = sys.argv[2]

p = '/ifs/loni/faculty/thompson/adni/gfleishm/ADNI2'
paths = glob.glob(p+'/*/icbm/'+pid+'/*.nii.gz')
paths.sort()
n = len(paths)

colors = np.array([
                    [1.0, 0.0, 0.0],    # Red
                    [0.0, 1.0, 0.0],    # Green
                    [0.0, 0.0, 1.0],    # Blue
                    [1.0, 1.0, 0.0],    # Yellow
                    [1.0, 0.0, 1.0],    # Magenta
                    [0.0, 1.0, 1.0],    # Cyan
                    [1.0, 1.0, 1.0]     # Grey
                        ])

fig = plt.figure('LA QC Image', figsize=(14, 9))

for p in range(n):

    plt.figtext(x=0.05, y=0.98-0.02*p, s=paths[p])
    img = nib.load(paths[p]).get_data().squeeze()
    img = img*(1.0/img.max())
    img = np.reshape(img, img.shape + (1,))*colors[p]

    fig.add_subplot(1, 3, 1)
    plt.imshow(np.rot90(img[110, :, :]), alpha=0.5)
    plt.axis('off')

    fig.add_subplot(1, 3, 2)
    plt.imshow(np.rot90(img[:, 110, :]), alpha=0.5)
    plt.axis('off')

    fig.add_subplot(1, 3, 3)
    plt.imshow(np.rot90(img[:, :, 110]), alpha=0.5)
    plt.axis('off')

plt.savefig(wPath)

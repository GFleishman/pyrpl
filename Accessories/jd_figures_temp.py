#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:25:12 2017

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

jd_filepath = sys.argv[1]
jd = nib.load(jd_filepath).get_data().squeeze()

fig = plt.figure()
fig.add_subplot(1, 3, 1)
plt.imshow(np.rot90(jd[110, :, :]))
plt.axis('off')
fig.add_subplot(1, 3, 2)
plt.imshow(np.rot90(jd[:, 110, :]))
plt.axis('off')
fig.add_subplot(1, 3, 3)
plt.imshow(np.rot90(jd[:, :, 110]))
plt.axis('off')
plt.colorbar()

plt.subplots_adjust(wspace=None, hspace=None)
fig.savefig(sys.argv[2])

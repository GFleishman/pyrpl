# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:56:36 2015

@author: gfleishman
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm

# grab inputs
pid = sys.argv[1]
wPath = sys.argv[2]

# Grab data
I0 = np.load(pid+'/I0.npy')
I1 = np.load(pid+'/I1.npy')
J = np.load(pid+'/J.npy')
dist = np.load(pid+'/dist.npy')
P0 = np.load(pid+'/P0.npy')

# Make figure
fig = plt.figure('IR QC Image', figsize=(14, 9))

fig.add_subplot(3, 3, 1)
plt.imshow(np.rot90(I0[:, :, 110]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(3, 3, 2)
plt.imshow(np.rot90(I1[:, :, 110]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(3, 3, 3)
plt.imshow(np.rot90(J[:, :, 110]), cmap=cm.gray)
plt.axis('off')

fig.add_subplot(3, 3, 4)
plt.imshow(np.rot90(I0[:, 110, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(3, 3, 5)
plt.imshow(np.rot90(I1[:, 110, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(3, 3, 6)
plt.imshow(np.rot90(J[:, 110, :]), cmap=cm.gray)
plt.axis('off')

fig.add_subplot(3, 3, 7)
plt.plot(range(len(dist[1])), dist[1])

fig.add_subplot(3, 3, 8)
plt.imshow(np.rot90(P0[:, :, 110]), cmap=cm.gray)
plt.axis('off')
plt.colorbar()
fig.add_subplot(3, 3, 9)
plt.imshow(np.rot90(P0[:, 110, :]), cmap=cm.gray)
plt.axis('off')
plt.colorbar()

# tighten up and save figure
plt.subplots_adjust(wspace=None, hspace=None)
fig.savefig(wPath)

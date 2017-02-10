# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 07:48:09 2015

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import transformer
import vcalc
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm

dr = sys.argv[1]
refPath = sys.argv[2]
wPath = sys.argv[3]

dist = np.load(dr + '/data_match.npy')

I0 = nib.load(dr + '/I0.nii.gz').get_data().squeeze()
I1 = nib.load(dr + '/I1.nii.gz').get_data().squeeze()
P0 = nib.load(dr + '/P0.nii.gz').get_data().squeeze()
J1 = nib.load(refPath).get_data().squeeze()

uf1 = np.empty(I0.shape + (3,))
for i in range(3):
    p = dr + '/uf1_' + str(i) + '.nii.gz'
    uf1[..., i] = nib.load(p).get_data().squeeze()

ub1 = np.empty(I0.shape + (3,))
for i in range(3):
    p = dr + '/ub1_' + str(i) + '.nii.gz'
    ub1[..., i] = nib.load(p).get_data().squeeze()

_t = transformer.transformer()
vox = np.array([1.0, 1.0, 1.0])
J0 = _t.applyTransform(J1, vox, uf1)

jd = np.linalg.det(vcalc.jacobian(uf1, vox))

fig = plt.figure(figsize=(14, 10))

fig.add_subplot(4, 4, 1)
plt.imshow(np.rot90(I0[110, :, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 2)
plt.imshow(np.rot90(J0[110, :, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 3)
plt.imshow(np.rot90(I1[110, :, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 4)
plt.imshow(np.rot90(J1[110, :, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 5)
plt.imshow(np.rot90(I0[:, 110, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 6)
plt.imshow(np.rot90(J0[:, 110, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 7)
plt.imshow(np.rot90(I1[:, 110, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 8)
plt.imshow(np.rot90(J1[:, 110, :]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 9)
plt.imshow(np.rot90(I0[:, :, 110]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 10)
plt.imshow(np.rot90(J0[:, :, 110]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 11)
plt.imshow(np.rot90(I1[:, :, 110]), cmap=cm.gray)
plt.axis('off')
fig.add_subplot(4, 4, 12)
plt.imshow(np.rot90(J1[:, :, 110]), cmap=cm.gray)
plt.axis('off')

fig.add_subplot(4, 4, 13)
plt.plot(range(len(dist[1])), dist[1])

fig.add_subplot(4, 4, 14)
plt.imshow(np.rot90(jd[110, :, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(4, 4, 15)
plt.imshow(np.rot90(jd[:, 110, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(4, 4, 16)
plt.imshow(np.rot90(jd[:, :, 110]))
plt.axis('off')
plt.colorbar()

plt.subplots_adjust(wspace=None, hspace=None)
fig.savefig(wPath)

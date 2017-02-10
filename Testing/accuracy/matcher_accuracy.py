# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:41:27 2014

@author: gfleishman
"""

import numpy as np
import matcher
import nibabel as nib
import matplotlib.pyplot as plt

pPath = '/Users/gfleishman/ScientificData/ADNI2b'
tmpPath = '/002_S_4171/002_S_4171_sc_ss.nii.gz'
#refPath = '/002_S_4171/002_S_4171_sc_ss.nii.gz'
#refPath = '/002_S_4171/NZ_002_S_4171_sc_ss.nii.gz'
#refPath = '/002_S_4171/002_S_4171_12mo_ss.nii.gz'
refPath = '/006_S_4449/006_S_4449_sc_ss.nii.gz'

tmp = nib.load(pPath+tmpPath).get_data().squeeze()
ref = nib.load(pPath+refPath).get_data().squeeze()

ref = ref*(1.0/np.mean(ref[ref != 0]))
tmp = tmp*(1.0/np.mean(tmp[tmp != 0]))

"""
m = matcher.matcher('SSD')
print m.dist(ref, tmp)
f = m.residual(ref, tmp)
fig = plt.figure(1, figsize=(12, 6))
fig.add_subplot(1, 3, 1)
plt.imshow(np.rot90(f[:, 98, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(1, 3, 2)
plt.imshow(np.rot90(f[:, 108, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(1, 3, 3)
plt.imshow(np.rot90(f[:, 118, :]))
plt.axis('off')
plt.colorbar()

m = matcher.matcher('CC')
print m.dist(ref, tmp)
f = m.residual(ref, tmp)
fig = plt.figure(2, figsize=(12, 6))
fig.add_subplot(1, 3, 1)
plt.imshow(np.rot90(f[:, 98, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(1, 3, 2)
plt.imshow(np.rot90(f[:, 108, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(1, 3, 3)
plt.imshow(np.rot90(f[:, 118, :]))
plt.axis('off')
plt.colorbar()

m = matcher.matcher('CCL')
print m.dist(ref, tmp)
f = m.residual(ref, tmp)
fig = plt.figure(3, figsize=(12, 6))
fig.add_subplot(1, 3, 1)
plt.imshow(np.rot90(f[:, 98, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(1, 3, 2)
plt.imshow(np.rot90(f[:, 108, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(1, 3, 3)
plt.imshow(np.rot90(f[:, 118, :]))
plt.axis('off')
plt.colorbar()
"""

m = matcher.matcher('MI')
print m.dist(ref, tmp)
f = m.residual(ref, tmp)
fig = plt.figure(4, figsize=(12, 6))
fig.add_subplot(1, 3, 1)
plt.imshow(np.rot90(f[:, 98, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(1, 3, 2)
plt.imshow(np.rot90(f[:, 108, :]))
plt.axis('off')
plt.colorbar()
fig.add_subplot(1, 3, 3)
plt.imshow(np.rot90(f[:, 118, :]))
plt.axis('off')
plt.colorbar()
plt.draw()
plt.show()

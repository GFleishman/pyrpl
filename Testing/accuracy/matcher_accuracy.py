# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:41:27 2014

@author: gfleishman
"""

import numpy as np
import pyrpl.image_tools.matcher as matcher
import nibabel as nib
import matplotlib.pyplot as plt
import pyrpl.image_tools.preprocessing as pp

pPath = '/Users/gfleishman/Desktop/temp_pyrpl_data'
tmpPath = '/moving.nii.gz'
#refPath = '/002_S_4171/002_S_4171_sc_ss.nii.gz'
#refPath = '/002_S_4171/NZ_002_S_4171_sc_ss.nii.gz'
#refPath = '/002_S_4171/002_S_4171_12mo_ss.nii.gz'
refPath = '/fixed.nii.gz'

tmp_img = nib.load(pPath+tmpPath)
tmp = tmp_img.get_data().squeeze()
ref_img = nib.load(pPath+refPath)
ref = ref_img.get_data().squeeze()

# if images are not already compressed and scaled
#tmp = pp.compress_intensity_range(tmp)
#tmp = pp.scale_intensity(tmp, mean=1.0)
#ref = pp.compress_intensity_range(ref)
#ref = pp.scale_intensity(ref, mean=1.0)


m = matcher.matcher('ssd', 11)
print m.dist(ref, tmp)
f = m.residual(ref, tmp)
f_img = nib.Nifti1Image(f, tmp_img.affine)
nib.save(f_img, '/Users/gfleishman/Desktop/temp_pyrpl_data/ssd.nii.gz')
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


m = matcher.matcher('gcc', 11)
print m.dist(ref, tmp)
f = m.residual(ref, tmp)
f_img = nib.Nifti1Image(f, tmp_img.affine)
nib.save(f_img, '/Users/gfleishman/Desktop/temp_pyrpl_data/gcc.nii.gz')
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


m = matcher.matcher('lcc', 11)
print m.dist(ref, tmp)
f = m.residual(ref, tmp)
f_img = nib.Nifti1Image(f, tmp_img.affine)
nib.save(f_img, '/Users/gfleishman/Desktop/temp_pyrpl_data/lcc.nii.gz')
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


m = matcher.matcher('MI', 256)
print m.dist(ref, tmp)
f = m.residual(ref, tmp)
f_img = nib.Nifti1Image(f, tmp_img.affine)
nib.save(f_img, '/Users/gfleishman/Desktop/temp_pyrpl_data/mi.nii.gz')
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

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 11:14:28 2015

@author: gfleishman
"""

import numpy as np
import nibabel as nib
import transformer
import sys

# get file paths
temp_mask_p = sys.argv[1]
vent_mask_p = sys.argv[2]
txm_p = sys.argv[3]
mdt_p = sys.argv[4]
temp_wp = sys.argv[5]
vent_wp = sys.argv[6]

# get transformation, masks, and mdt
temp_mask = nib.load(temp_mask_p).get_data().squeeze()
vent_mask = nib.load(vent_mask_p).get_data().squeeze()
txm = np.empty(temp_mask.shape + (3,))
for i in range(3):
    txm[..., i] = nib.load(txm_p + str(i) + '.nii.gz').get_data().squeeze()
mdt = nib.load(mdt_p).get_data().squeeze()

# define image voxel size and make a transformer
vox = np.array([1., 1., 1.])
t = transformer.transformer()

# transform temporal mask, keep interpolants above threshold
temp_mask = t.applyTransform(temp_mask, vox, txm)
temp_mask[temp_mask >= 0.95] = 1.0
temp_mask[temp_mask < 0.95] = 0.0

# transform ventricle mask, keep interpolants above threshold
vent_mask = t.applyTransform(vent_mask, vox, txm)
vent_mask[vent_mask >= 0.9] = 1.0
vent_mask[vent_mask < 0.9] = 0.0

# intersect masks, remove intersection from temporal mask
inter = temp_mask*vent_mask
temp_mask = temp_mask - inter

# remove voxels outside brain region
temp_mask[mdt <= 0.8] = 0.0
vent_mask[mdt == 0.0] = 0.0

# write out results as nifti files
vent_mask = nib.Nifti1Image(vent_mask, np.eye(4))
nib.save(vent_mask, vent_wp)
temp_mask = nib.Nifti1Image(temp_mask, np.eye(4))
nib.save(temp_mask, temp_wp)

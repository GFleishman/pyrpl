# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:11:59 2015

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import transformer

# get file paths
jdp_root = sys.argv[1]
txm_root = sys.argv[2]
wPath = sys.argv[3]

# image details, transformer for later
sh = (220, 220, 220)
vox = np.array([1., 1., 1.])
_t = transformer.transformer()

# grab the data
jd = nib.load(jdp_root + '/jd.nii.gz').get_data().squeeze()

# grab the transformation to mdt coordinates
txm = np.empty(sh + (3,))
for i in range(3):
    p = txm_root + str(i) + '.nii.gz'
    txm[..., i] = nib.load(p).get_data().squeeze()

# Apply txm to data
jd = _t.applyTransform(jd, vox, txm)

# save results
jd = nib.Nifti1Image(jd, np.eye(4))
nib.save(jd, wPath + '/jd_mdt.nii.gz')

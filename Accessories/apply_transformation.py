# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:11:59 2015

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import pyrpl.image_tools.transformer as transformer

# get file paths
img_root = sys.argv[1]
txm_root = sys.argv[2]
wPath = sys.argv[3]

# image details, transformer for later
sh = (220, 220, 220)
vox = np.array([1., 1., 1.])
_t = transformer.transformer()

# grab the data
img = nib.load(img_root).get_data().squeeze()

# grab the transformation to mdt coordinates
txm = np.empty(sh + (3,))
for i in range(3):
    p = txm_root + str(i) + '.nii.gz'
    txm[..., i] = nib.load(p).get_data().squeeze()

# TEMP: these txms are old, need to have position array added
txm += _t.position_array(sh, vox)

# Apply txm to data
img = _t.applyTransform(img, vox, txm)

# save results
img = nib.Nifti1Image(img, np.eye(4))
nib.save(img, wPath)

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:00:38 2015

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import pyrpl.image_tools.vcalc as vcalc

path = sys.argv[1]
wPath = sys.argv[2]
vox = np.array([1., 1., 1.])

uf1 = np.empty((220, 220, 220, 3))
for i in range(3):
    p = path + str(i) + '.nii.gz'
    uf1[..., i] = nib.load(p).get_data().squeeze()

jd = np.linalg.det(vcalc.jacobian(uf1, vox))
jd = nib.Nifti1Image(jd, np.eye(4))
nib.save(jd, wPath)

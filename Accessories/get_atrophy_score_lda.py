# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:09:52 2016

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib

mask_path = sys.argv[1]
jd_path = sys.argv[2]
w_path = sys.argv[3]

mask = nib.load(mask_path).get_data().squeeze()
mask = mask/np.sum(mask)
jd = np.load(jd_path)

out = np.sum(mask*jd)

np.save(w_path, out)
print str(out)

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
jd = nib.load(jd_path).get_data().squeeze()
jd[np.isnan(jd)] = 1.0


out = np.sum(mask*jd)/np.sum(mask)
out = (1.0 - out)*100.0

np.save(w_path, out)
print str(out)

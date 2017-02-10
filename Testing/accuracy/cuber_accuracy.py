# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:42:53 2015

@author: gfleishman
"""

import numpy as np
import nibabel as nib
import cuber

"""ICBM Template"""
pPath = '/Users/gfleishman/Desktop/Preprocess4/ICBM_SS'
tmpPath = '/ICBM_SS.hdr'    
tmp = nib.load(pPath+tmpPath).get_data().squeeze()
tmp = tmp*(1.0/tmp.max())   
ref = np.copy(tmp)

print ref.shape
print tmp.shape
refn = cuber.cube(ref, (1.0, 1.0, 1.0))
tmpn = cuber.cube(tmp, (1.0, 1.0, 1.0))
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print refn.shape
print tmpn.shape

import matplotlib.pyplot as plt
fig = plt.figure('cube check', figsize=(8,6))
fig.add_subplot(2,2,1)
plt.imshow(ref[...,95])
fig.add_subplot(2,2,2)
plt.imshow(refn[...,95])
fig.add_subplot(2,2,3)
plt.imshow(tmp[...,95])
fig.add_subplot(2,2,4)
plt.imshow(tmpn[...,95])

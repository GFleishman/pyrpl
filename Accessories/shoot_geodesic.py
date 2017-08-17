# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:30:32 2016

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import geodesic_regression_in_diffeomorphisms as grid

tPath = sys.argv[1]
mPath = sys.argv[2]
time = float(sys.argv[3])
h = int(sys.argv[4])
outdir = sys.argv[5]

# load the template and momenta images
T_img = nib.load(tPath)
T = T_img.get_data().squeeze()
T = T*(1.0/np.mean(T[T != 0]))
P = nib.load(mPath).get_data().squeeze()

# establish particulars for shooting template
params = {
        'vox': np.array([1.0, 1.0, 1.0]),
        'its': [1],
        'res': [T.shape],
        'rat': 2.0,
        'h': h,
        'a': 1.0,
        'b': 0.0,
        'c': 0.1,
        'd': 2.0,
        'mType': 'CCL',
        'rType': 'differential'
        }

# construct grdc object
J = np.array([T, T])
tm = np.array([0.0, time])

# initialize shooter and regularizer
_gs = grid.geodesic_regression_in_diffeomorphisms(J, tm, params)
_gs.resample(params['res'])
_gs.dc.P[0] = P

# integrate
_gs.solveForward(grdc)

# write out endpoint of path
img = nib.Nifti1Image(grdc.uf[-1, ..., 0], T_img.affine)
nib.save(img, outdir + '/uf1_0.nii.gz')
img = nib.Nifti1Image(grdc.uf[-1, ..., 1], T_img.affine)
nib.save(img, outdir + '/uf1_1.nii.gz')
if grdc.d == 3:
    img = nib.Nifti1Image(grdc.uf[-1, ..., 2], T_img.affine)
    nib.save(img, outdir + '/uf1_2.nii.gz')

img = nib.Nifti1Image(grdc.ub[-1, ..., 0], T_img.affine)
nib.save(img, outdir + '/ub1_0.nii.gz')
img = nib.Nifti1Image(grdc.ub[-1, ..., 1], T_img.affine)
nib.save(img, outdir + '/ub1_1.nii.gz')
if grdc.d == 3:
    img = nib.Nifti1Image(grdc.ub[-1, ..., 2], T_img.affine)
    nib.save(img, outdir + '/ub1_2.nii.gz')

img = nib.Nifti1Image(grdc.Ifr[-1], T_img.affine)
nib.save(img, outdir + '/I1.nii.gz')

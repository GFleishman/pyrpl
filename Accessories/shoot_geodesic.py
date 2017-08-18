# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:30:32 2016

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import pyrpl.image_tools.preprocessing as pp
import pyrpl.models.gsid as gsid

# grab inputs
tPath = sys.argv[1]
mPath = sys.argv[2]
time = float(sys.argv[3])
h = int(sys.argv[4])
outdir = sys.argv[5]

# load the template and momenta images
T_img = nib.load(tPath)
T = T_img.get_data().squeeze()
T = pp.scale_intensity(T, mean=1.0)
P = nib.load(mPath).get_data().squeeze()

# establish parameters for shooting template
params = {
        'vox': np.array([1.0, 1.0, 1.0]),
        'its': [1],
        'res': [T.shape],
        'rat': 1.0,
        'h': h,
        'a': 1.0,
        'b': 0.0,
        'c': 0.05,
        'd': 2.0,
        'mType': 'SSD',
        'rType': 'differential'
        }

# construct model object
J = np.array([T, T])
tm = np.array([0.0, time])
_gs = gsid.gsid(J, tm, params)

# resample to correct resolution and set initial conditions
_gs.resample(params['res'])
_gs.dc.P[0] = P

# shoot geodesic
_gs.evaluate()

# write out endpoint of path
img = nib.Nifti1Image(_gs.dc.uf[-1, ..., 0], T_img.affine)
nib.save(img, outdir + '/uf1_0.nii.gz')
img = nib.Nifti1Image(_gs.dc.uf[-1, ..., 1], T_img.affine)
nib.save(img, outdir + '/uf1_1.nii.gz')
if _gs.dc.d == 3:
    img = nib.Nifti1Image(_gs.dc.uf[-1, ..., 2], T_img.affine)
    nib.save(img, outdir + '/uf1_2.nii.gz')

img = nib.Nifti1Image(_gs.dc.ub[-1, ..., 0], T_img.affine)
nib.save(img, outdir + '/ub1_0.nii.gz')
img = nib.Nifti1Image(_gs.dc.ub[-1, ..., 1], T_img.affine)
nib.save(img, outdir + '/ub1_1.nii.gz')
if _gs.dc.d == 3:
    img = nib.Nifti1Image(_gs.dc.ub[-1, ..., 2], T_img.affine)
    nib.save(img, outdir + '/ub1_2.nii.gz')

img = nib.Nifti1Image(_gs.dc.Ifr[-1], T_img.affine)
nib.save(img, outdir + '/I1.nii.gz')

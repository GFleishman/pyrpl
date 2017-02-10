# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:41:07 2016

@author: gfleishman
"""

import sys
import numpy as np
import nibabel as nib
import scipy.stats as stats

sample1_path = sys.argv[1]
sample2_path = sys.argv[2]
wPath = sys.argv[3]
sh = (220, 220, 220)  # Image resolution: magic number

sample1 = open(sample1_path, 'r').read().split('\n')
sample1 = sample1[:-1]
sample2 = open(sample2_path, 'r').read().split('\n')
sample2 = sample2[:-1]

# AD
mean1 = np.zeros(sh)
var1 = np.zeros(sh)
n1 = float(len(sample1))
for s in sample1:
    print s
    img = nib.load(s)
    mean1 += img
    var1 += img**2

mean1 *= 1.0/n1
var1 *= 1.0/n1
var1 -= mean1**2

# Control
mean2 = np.zeros(sh)
var2 = np.zeros(sh)
n2 = float(len(sample2))
for s in sample2:
    print s
    img = np.load(s)
    mean2 += img
    var2 += img**2

mean2 *= 1.0/n2
var2 *= 1.0/n2
var2 -= mean2**2

# compute p-values
t = abs(mean1 - mean2)/(var1/n1 + var2/n2)**0.5
den = (var1/n1)**2/(n1 - 1) + (var2/n2)**2/(n2 - 1)
df = (var1/n1 + var2/n2)**2/den
pvals = 1.0 - stats.t.cdf(t, df)

# Only want p-values for voxels with atrophy
pvals[mean1 >= 1.0] = 1.0

#np.save(wPath, pvals)
np.save(wPath + '/t_scores.npy', t)
np.save(wPath + '/mean_jd.npy', mean1)

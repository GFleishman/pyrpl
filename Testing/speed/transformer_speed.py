"""
Author: Greg M. Fleishman

Description: A test of the speed of a 2D image transformation

Dependencies: NumPy, timeit, pyrt.regTools.transformer
"""

import numpy as np
import timeit


setup = """
import numpy as np
import transformer

#Create function to test
resx = 128
resy = 128

vox = [1.0/resx, 1.0/resy]
rngx = np.arange(0, resx)*vox[0]
rngy = np.arange(0, resy)*vox[1]
x, y = np.meshgrid(rngx, rngy, indexing='ij')
R = 2*np.pi
sn = np.sin
cs = np.cos

img = sn(R*x) + cs(R*y) - sn(R*y)*cs(R*x)   #The image

ttlx = cs(R*x)*sn(R*y)
ttly = sn(R*x)*cs(R*y)

#numerically transform image
_t = transformer.transformer()
_t._initialize(img.shape)
txm = np.empty(img.shape + (len(img.shape),))
txm[...,0] = ttlx
txm[...,1] = ttly
"""

# time transformation
tr = (timeit.Timer("_t.applyTransform(img, vox, txm)", setup).repeat(100, 1))
print "minimum transformation computation time: " + str(np.min(tr))

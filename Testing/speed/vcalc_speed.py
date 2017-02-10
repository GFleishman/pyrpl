"""
Author: Greg M. Fleishman

Description: A test of the speed of vcalc functions

Dependencies: NumPy, timeit, pyrt.regTools.vcalc
"""

import numpy as np
import timeit


setup = """
import numpy as np
import vcalc

#Create test img/function
xres = np.random.rand()*128.0 + 128.0
xres = np.floor(xres)

yres = np.random.rand()*128.0 + 128.0
yres = np.floor(yres)

resf = np.array([xres, yres])
res = np.array([int(xres), int(yres)])

rngx = np.arange(0, res[0])/resf[0]
rngy = np.arange(0, res[1])/resf[1]
vox = np.ones(2)/resf

x, y = np.meshgrid(rngx, rngy, indexing='ij')
R = 2*np.pi
sn = np.sin
cs = np.cos

img = sn(R*x) + cs(R*y) - sn(R*y)*cs(R*x)
"""

# time gradients
tg = (timeit.Timer("vcalc.gradient(img, vox)", setup).repeat(300, 1))
print "minimum gradient computation time: " + str(np.min(tg))

# add gradient calc to steup
setup = setup + """
g = vcalc.gradient(img, vox)
"""

# time jacobian
tj = (timeit.Timer("vcalc.jacobian(g, vox);", setup).repeat(300, 1))
print "minimum jacobian computation time: " + str(np.min(tj))

# time divergence
td = (timeit.Timer("vcalc.divergence(g, vox);", setup).repeat(300, 1))
print "minimum divergence computation time: " + str(np.min(td))

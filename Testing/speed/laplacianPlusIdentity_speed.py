import numpy as np
import timeit



setup = """
import numpy as np
import regularizer

#Create test img/function
xres = np.random.rand()*128.0 + 128.0
xres = np.floor(xres)

yres = np.random.rand()*128.0 + 128.0
yres = np.floor(yres)

resf = np.array([xres, yres])
res = np.array([int(xres), int(yres)])

rngx = np.arange(0, res[0])/resf[0]
rngy = np.arange(0, res[1])/resf[1]

x, y = np.meshgrid(rngx, rngy, indexing='ij')
R = 2*np.pi
sn = np.sin
cs = np.cos

img = sn(R*x) + cs(R*y) - sn(R*y)*cs(R*x)
img = np.reshape(img, img.shape + (1,))
img = np.concatenate((img,img), axis=-1)

#Initialize regularizer
m = 0.01
l = 1.0
k = 1.0
_r = regularizer._LaplacianPlusIdentity()
_r._initialize(m, l, (int(xres),int(yres)), k)
"""



#time regularizer
tr = (timeit.Timer("_r.regularize(img)", setup).repeat(20,20))
print np.mean(tr)